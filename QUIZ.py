import streamlit as st
import pandas as pd
import numpy as np
import cv2
import io
import os
import re
import pdfplumber
import base64  # 💡 為了將圖片傳給 GPT-4o，需要 base64 編碼
from PIL import Image
from pdf2image import convert_from_bytes
from google.cloud import vision
from google.api_core.client_options import ClientOptions

# 1. 引入最新的 Gemini GenAI SDK (負責後續結構化批改)
from google.genai import Client as GeminiClient
from google.genai import types
from pydantic import BaseModel, Field

# 2. 引入 OpenAI SDK (負責精準區域 OCR)
from openai import OpenAI

# --- Define Pydantic Schema for Gemini Structured Output ---

class GradingResult(BaseModel):
    score: float = Field(description="根據標準答案與學生作答給予的得分，不可以超過該題的最高配分")
    reason: str = Field(description="詳細的評分理由與針對學生作答的講評")

# --- 1. 核心影像處理與區域文字提取邏輯 ---

def order_points_robust(pts):
    """強健的點排序：左上, 右上, 右下, 左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # tl
    rect[2] = pts[np.argmax(s)]    # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def enhance_for_ocr(img_np):
    """
    優化後的影像強化邏輯
    """
    if img_np is None or img_np.size == 0:
        return None    
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 5, 7, 21)    
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def has_visible_content_in_crop(img_np, y_start, y_end, threshold_ratio=0.005):
    """
    針對掃描檔的盲區內容偵測
    """
    h, w, _ = img_np.shape
    y_start = max(0, int(y_start))
    y_end = min(h, int(y_end))
    
    if (y_end - y_start) < 10:
        return False
        
    crop_roi = img_np[y_start:y_end, 0:w]
    gray = cv2.cvtColor(crop_roi, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    non_zero_count = cv2.countNonZero(binary)
    total_pixels = binary.size
    black_pixel_ratio = non_zero_count / total_pixels
    
    return black_pixel_ratio > threshold_ratio


def fix_ocr_text_with_llm(raw_text, openai_client):
    """
    OCR結果修復
    只允許：
    1. 修正明顯OCR錯字
    2. 修正排版
    3. 合併斷行

    不允許：
    1. 補充內容
    2. 推論內容
    3. 增加公式
    """

    if not raw_text.strip():
        return ""

    prompt = f"""
以下是OCR結果，可能有排版錯誤。

請嚴格遵守：

1. 只能修正明顯OCR錯字
2. 只能修正排版
3. 可以合併被切斷的句子
4. 不可以補充任何內容
5. 不可以推測遺失內容
6. 不可以增加不存在的文字
7. 保留原意

只輸出修正後內容：

{raw_text}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return raw_text



def detect_and_ocr_boxes(pdf_bytes, openai_client, min_w=400, min_h=100, need_debug=False):
    """
    結合 OpenCV 定位與 pdfplumber/GPT-4o 的混合文字提取。
    💡 調整為使用 openai_client 來執行無情且精準的框選 OCR。
    """
    images = convert_from_bytes(pdf_bytes, dpi=350)
    all_text_results = []
    debug_images = []

    page_blocks = {} 
    page_cv_images = {} 

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p_idx, img in enumerate(images):
            img_np = np.array(img.convert("RGB"))
            page_cv_images[p_idx] = img_np.copy()
            h_img, w_img, _ = img_np.shape
            
            page_plumber = pdf.pages[p_idx]
            w_pdf, h_pdf = page_plumber.width, page_plumber.height
            
            scale_x = w_pdf / w_img
            scale_y = h_pdf / h_img

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 21, 18)

            h_k = max(60, int(min_w * 0.5)) 
            v_k = max(30, int(min_h * 0.6))
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_k, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k))
            
            detected_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            detected_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            detected_lines = cv2.add(detected_h, detected_v)

            contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                if w < h:
                    w, h = h, w
                    angle += 90
                
                if abs(angle) < 5 or abs(angle - 180) < 5: angle = 0
                
                if w >= min_w and h >= min_h:
                    box = cv2.boxPoints(((cx, cy), (w, h), angle))
                    box = box.astype(int)
                    candidates.append(((cx, cy), (w, h), angle, box))

            candidates = sorted(candidates, key=lambda x: x[0][1])
            page_blocks[p_idx] = []

            for (cx, cy), (w, h), angle, box_points in candidates:
                x_min, y_min = np.min(box_points, axis=0)
                x_max, y_max = np.max(box_points, axis=0)
                
                pdf_bbox = (
                    x_min * scale_x + 3, 
                    y_min * scale_y + 3, 
                    x_max * scale_x - 3, 
                    y_max * scale_y - 3
                )
                
                extracted_text = None
                try:
                    crop = page_plumber.within_bbox(pdf_bbox)
                    extracted_text = crop.extract_text()
                except Exception:
                    extracted_text = None

                # --- 🔀 核心修改：切換至 GPT-4o 執行 OCR 區塊 ---
                if extracted_text and extracted_text.strip():
                    text = extracted_text.strip()
                else:
                    rect_pts = order_points_robust(box_points)
                    dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
                    warped = cv2.warpPerspective(img_np, M, (int(w), int(h)))
                    
                    pad = 10
                    if w > pad*2 and h > pad*2:
                        warped = warped[pad:int(h)-pad, pad:int(w)-pad]
                    
                    cropped_img = Image.fromarray(warped)

                    try:
                        buf = io.BytesIO()
                        cropped_img.save(buf, format="JPEG")
                        image_bytes = buf.getvalue()
                        
                        # 將圖片轉成 GPT-4o 所需的 Base64 字串
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # 💡 完美融入您的嚴格 OCR 規則
                        ocr_prompt = (
                            "請將框選出的文字精準輸出給我。嚴格遵守以下規則：\n"
                            "1. 只輸出圖片框框中實際存在的文字，絕對不要新增任何框框中沒有出現的解釋、摘要或延伸說明。\n"
                            "2. 不要包含任何前後引言、客套話或補充說明（例如：『以下是辨識結果』），直接輸出辨識內容即可。\n"
                            "3. 還有不要把中文識別成英文，英文識別成中文。"
                        )
                        
                        # 呼叫 GPT-4o
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": ocr_prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            temperature=0.0  # 設為 0.0 確保輸出最精確、不發散
                        )
                        raw_text = (
                            response.choices[0].message.content.strip()
                            if response.choices[0].message.content
                            else ""
                        )

                        text = fix_ocr_text_with_llm(
                            raw_text,
                            openai_client
                        )
                    
                    except Exception as e:
                        st.warning(f"區域 GPT-4o OCR 發生錯誤: {e}")
                        text = ""
                # --- 核心 OCR 區塊修改結束 ---

                page_blocks[p_idx].append({
                    "text": text,
                    "bbox": pdf_bbox,
                    "img_box": (x_min, y_min, x_max, y_max)
                })
                
                if need_debug:
                    cv2.drawContours(img_np, [box_points], 0, (0, 255, 0), 3)

            if need_debug:
                debug_images.append(Image.fromarray(img_np))

        # --- 跨頁表格合併邏輯 ---
        total_pages = len(images)
        for p_idx in range(total_pages):
            if p_idx > 0 and len(page_blocks[p_idx - 1]) > 0 and len(page_blocks[p_idx]) > 0:
                
                prev_page_plumber = pdf.pages[p_idx - 1]
                curr_page_plumber = pdf.pages[p_idx]
                
                last_block_prev_page = page_blocks[p_idx - 1][-1]
                first_block_curr_page = page_blocks[p_idx][0]
                
                text_below_last_table = ""
                text_above_first_table = ""
                try:
                    bottom_crop = prev_page_plumber.crop((0, last_block_prev_page["bbox"][3], prev_page_plumber.width, prev_page_plumber.height))
                    text_below_last_table = bottom_crop.extract_text() or ""
                    
                    top_crop = curr_page_plumber.crop((0, 0, curr_page_plumber.width, first_block_curr_page["bbox"][0]))
                    text_above_first_table = top_crop.extract_text() or ""
                except Exception:
                    pass
                
                has_digital_text_between = bool(text_below_last_table.strip() or text_above_first_table.strip())
                
                prev_img_np = page_cv_images[p_idx - 1]
                curr_img_np = page_cv_images[p_idx]
                
                _, _, _, last_y_max = last_block_prev_page["img_box"]
                _, first_y_min, _, _ = first_block_curr_page["img_box"]
                
                has_scanned_content_below = has_visible_content_in_crop(prev_img_np, last_y_max, prev_img_np.shape[0])
                has_scanned_content_above = has_visible_content_in_crop(curr_img_np, 0, first_y_min)
                
                has_image_content_between = has_scanned_content_below or has_scanned_content_above
                
                if (not has_digital_text_between) and (not has_image_content_between):
                    last_block_prev_page["text"] += "\n" + first_block_curr_page["text"]
                    first_block_curr_page["is_merged_child"] = True

        for p_idx in range(total_pages):
            for block in page_blocks[p_idx]:
                if not block.get("is_merged_child", False):
                    all_text_results.append(block["text"])

    return all_text_results, debug_images

# --- 2. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 評分系統", layout="wide")
    st.title("📑 全自動考卷批改工作台 (GPT-4o OCR + Gemini 2.0 批改版)")

    if "debug_images" not in st.session_state:
        st.session_state.debug_images = []
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["題目", "問題內容", "學生作答", "標準答案", "配分", "得分", "AI 評分理由"])

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 💡 新增 OpenAI 金鑰檢查
    
    if not GOOGLE_API_KEY:
        st.error("❌ 環境變數中找不到 GOOGLE_API_KEY，請確認 Streamlit Secrets 設定。")
        st.stop()
        
    if not OPENAI_API_KEY:
        st.error("❌ 環境變數中找不到 OPENAI_API_KEY，請確認 Streamlit Secrets 設定。")
        st.stop()

    try:
        # 初始化兩個大模型 Client 
        gemini_client = GeminiClient() 
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"💥 初始化 API 失敗，請檢查環境變數與憑證：{e}")
        return

    with st.sidebar:
        st.header("📥 上傳 PDF 來源")
        pdf_q = st.file_uploader("1. 問題內容 PDF", type="pdf")
        pdf_s = st.file_uploader("2. 學生作答 PDF", type="pdf")
        pdf_a = st.file_uploader("3. 標準答案 PDF", type="pdf")
        pdf_p = st.file_uploader("4. 配分 PDF", type="pdf")
        
        if st.button("🚀 開始全自動解析") and pdf_q and pdf_s and pdf_a and pdf_p:
            with st.spinner("正在執行精準座標對齊與 GPT-4o 視覺提取..."):
                q_bytes = pdf_q.read()
                s_bytes = pdf_s.read()
                a_bytes = pdf_a.read()
                p_bytes = pdf_p.read()

                # 💡 將 openai_client 傳入解析函式
                q_texts, _ = detect_and_ocr_boxes(q_bytes, openai_client)
                s_texts, s_debug_imgs = detect_and_ocr_boxes(s_bytes, openai_client, need_debug=True)
                a_texts, _ = detect_and_ocr_boxes(a_bytes, openai_client)
                p_texts, _ = detect_and_ocr_boxes(p_bytes, openai_client)
                
                num_questions = max(len(q_texts), len(s_texts), len(a_texts), len(p_texts))
                
                data = []
                for i in range(num_questions):
                    data.append({
                        "題目": f"第 {i+1} 題",
                        "問題內容": q_texts[i] if i < len(q_texts) else "",
                        "學生作答": s_texts[i] if i < len(s_texts) else "",
                        "標準答案": a_texts[i] if i < len(a_texts) else "",
                        "配分": p_texts[i] if i < len(p_texts) else "10",
                        "得分": 0.0,
                        "AI 評分理由": ""
                    })
                
                st.session_state.df = pd.DataFrame(data)
                st.session_state.debug_images = s_debug_imgs
                st.success(f"解析完成！共處理 {num_questions} 個題目區塊。")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📝 結構化評分表")
        
        if "main_editor" not in st.session_state:
            st.session_state.main_editor = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}

        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            height=600,
            key="my_data_editor"
        )
        
        st.session_state.df = edited_df

        try:
            current_score = pd.to_numeric(st.session_state.df["得分"]).sum()
        except Exception:
            current_score = 0

        btn_col, score_col = st.columns([1, 1])

        with btn_col:
            run_ai = st.button("🤖 執行 Gemini 自動批改", use_container_width=True)
        
        with score_col:
            st.markdown(f"### 🎯 目前總分：{current_score:.1f} / 100.0")

        # --- 執行 AI 批改邏輯 ---
        if run_ai:
            temp_df = st.session_state.df.copy()
            num_rows = len(temp_df)
            
            status_text = st.empty()
            
            for index, row in temp_df.iterrows():
                status_text.markdown(f"⏳ **Gemini Pro 正在批改第 {index + 1} / {num_rows} 題...**")
                
                if row["學生作答"]:
                    prompt = (
                        f"你是一位專業老師。此考卷採取「總分 100 分制」，請根據以下單題資訊進行精確評分：\n"
                        f"問題：{row['問題內容']}\n"
                        f"標準答案：{row['標準答案']}\n"
                        f"學生回答：{row['學生作答']}\n"
                        f"【重要】此題的最高配分（滿分）為：{row['配分']} 分。\n"
                        f"請評估學生的回答完整度與正確性，給予 0 到 {row['配分']} 之間的合理分數（可為小數）。\n"
                        f"你給出的分數「絕對不可以」超過該題的最高配分。\n"
                    )
                    try:
                        response = gemini_client.models.generate_content(
                            model='gemini-2.0-pro-exp-0205',  
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=GradingResult,
                                temperature=0.2, 
                            ),
                        )
                        
                        result: GradingResult = response.parsed
                        temp_df.at[index, "得分"] = float(result.score)
                        temp_df.at[index, "AI 評分理由"] = result.reason
                        
                    except Exception as e:
                        st.warning(f"第 {index+1} 題評分錯誤: {e}")
            
            status_text.empty()
            st.session_state.df = temp_df
            st.success("🎉 全數批改完成！")
            st.rerun()
            
    with col2:
        st.subheader("🔍 框選對齊視覺檢查")
        if st.session_state.debug_images:
            for i, img in enumerate(st.session_state.debug_images):
                st.image(img, caption=f"偵測區域預覽 - 頁面 {i+1}")
        else:
            st.info("暫無視覺檢查影像，請先上傳 PDF 並點擊開始全自動解析。")

if __name__ == "__main__":
    main()
