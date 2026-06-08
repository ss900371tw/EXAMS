import streamlit as st
import pandas as pd
import numpy as np
import cv2
import io
import os
import re
import pdfplumber
from PIL import Image
from pdf2image import convert_from_bytes
from google.cloud import vision
from google.api_core.client_options import ClientOptions

# 1. 引入最新的 Gemini GenAI SDK
from google.genai import Client
from google.genai import types
from pydantic import BaseModel, Field

# --- Define Pydantic Schema for Gemini Structured Output ---
class GradingResult(BaseModel):
    score: float = Field(description="根據標準答案與學生作答給予的得分，不可以超過最高分")
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
    影像強化邏輯：轉灰階 -> CLAHE 對比強化 -> 非局部均值去噪 (Denoising)
    """
    if img_np is None or img_np.size == 0:
        return None    
    # 1. 轉灰階
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # 2. 增加對比度 (CLAHE) - 讓模糊的字體與背景對比更強烈
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # 3. 去噪處理 (Denoising) - 替換原本的銳利化
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)    
    return denoised


def has_visible_content_in_crop(img_np, y_start, y_end, threshold_ratio=0.005):
    """
    針對掃描檔的盲區內容偵測：
    透過二值化分析特定 Y 軸區間內是否存在非空白內容（文字或符號）。
    - threshold_ratio: 黑色像素佔該區塊總面積的比例超過多少視為「有文字內容」
    """
    h, w, _ = img_np.shape
    
    # 安全範圍限縮，避免陣列越界
    y_start = max(0, int(y_start))
    y_end = min(h, int(y_end))
    
    # 如果裁剪區間過小，視為無阻隔文字
    if (y_end - y_start) < 10:
        return False
        
    crop_roi = img_np[y_start:y_end, 0:w]
    gray = cv2.cvtColor(crop_roi, cv2.COLOR_RGB2GRAY)
    
    # 使用大津二值化將文字（深色）與背景（白色）分離
    # 為了方便計算，轉為反轉二值化（文字變白色像素 255，背景變 0）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 計算非零像素（即文字內容）的比例
    non_zero_count = cv2.countNonZero(binary)
    total_pixels = binary.size
    black_pixel_ratio = non_zero_count / total_pixels
    
    # 超過閾值代表中間夾有題目文字或線條，兩者不應合併
    return black_pixel_ratio > threshold_ratio


def detect_and_ocr_boxes(pdf_bytes, vision_client, min_w=200, min_h=100, need_debug=False):
    """
    結合 OpenCV 定位與 pdfplumber/Google Vision 的混合文字提取。
    具備「跨頁表格文字連續性判定」機制（支援掃描檔與非掃描檔）。
    """
    images = convert_from_bytes(pdf_bytes, dpi=350)
    all_text_results = []
    debug_images = []

    # 用來暫存每一頁解析出來的區塊資訊與原始影像
    page_blocks = {} 
    page_cv_images = {} # 存放各頁的原始 ndarray 用於跨頁 OpenCV 檢測

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p_idx, img in enumerate(images):
            img_np = np.array(img.convert("RGB"))
            page_cv_images[p_idx] = img_np.copy() # 備份原始影像
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

            # 依 Y 軸由上到下排序
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
                    warped_enhanced = enhance_for_ocr(warped)
                    cropped_img = Image.fromarray(warped_enhanced)

                    buf = io.BytesIO()
                    cropped_img.save(buf, format="JPEG")
                    image_context = vision.ImageContext(language_hints=["zh-Hant", "en"])
                    vision_img = vision.Image(content=buf.getvalue())

                    response = vision_client.document_text_detection(
                        image=vision_img, 
                        image_context=image_context
                    )
                    text = response.full_text_annotation.text.strip() if response.full_text_annotation.text else ""

                page_blocks[p_idx].append({
                    "text": text,
                    "bbox": pdf_bbox,
                    "img_box": (x_min, y_min, x_max, y_max) # 保存原圖影像座標以利跨頁對比
                })
                
                if need_debug:
                    cv2.drawContours(img_np, [box_points], 0, (0, 255, 0), 3)

            if need_debug:
                debug_images.append(Image.fromarray(img_np))

        # --- 跨頁表格合併核心邏輯 (支援掃描檔雙重檢查) ---
        total_pages = len(images)
        for p_idx in range(total_pages):
            if p_idx > 0 and len(page_blocks[p_idx - 1]) > 0 and len(page_blocks[p_idx]) > 0:
                
                prev_page_plumber = pdf.pages[p_idx - 1]
                curr_page_plumber = pdf.pages[p_idx]
                
                last_block_prev_page = page_blocks[p_idx - 1][-1]
                first_block_curr_page = page_blocks[p_idx][0]
                
                # --- 測試 1: 數位型 PDF 判定法 (pdfplumber 區域文字提取) ---
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
                
                # --- 測試 2: 掃描檔 PDF 判定法 (OpenCV 盲區像素投影檢測) ---
                prev_img_np = page_cv_images[p_idx - 1]
                curr_img_np = page_cv_images[p_idx]
                
                # 取得原圖影像的 y1 (上邊界) 和 y2 (下邊界)
                _, _, _, last_y_max = last_block_prev_page["img_box"]
                _, first_y_min, _, _ = first_block_curr_page["img_box"]
                
                # 檢查前一頁表格下方到頁尾、當前頁開頭到第一個表格上方是否有可見圖文
                has_scanned_content_below = has_visible_content_in_crop(prev_img_np, last_y_max, prev_img_np.shape[0])
                has_scanned_content_above = has_visible_content_in_crop(curr_img_np, 0, first_y_min)
                
                has_image_content_between = has_scanned_content_below or has_scanned_content_above
                
                # --- 綜合判定決定是否合併 ---
                # 只有當「數位文字檢測」與「影像內容檢測」同時回傳乾淨空白時，才允許跨頁表格合併
                if (not has_digital_text_between) and (not has_image_content_between):
                    last_block_prev_page["text"] += "\n" + first_block_curr_page["text"]
                    first_block_curr_page["is_merged_child"] = True

        # --- 重新打包成最終的文字列表 ---
        for p_idx in range(total_pages):
            for block in page_blocks[p_idx]:
                if not block.get("is_merged_child", False):
                    all_text_results.append(block["text"])

    return all_text_results, debug_images


# --- 2. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 評分系統", layout="wide")
    st.title("📑 全自動考卷批改工作台 (Gemini Pro 強化版)")

    # 初始化 session_state
    if "debug_images" not in st.session_state:
        st.session_state.debug_images = []
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["題目", "問題內容", "學生作答", "標準答案", "配分", "得分", "AI 評分理由"])

    # --- API 客戶端初始化 ---
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    if not GOOGLE_API_KEY:
        st.error("❌ 環境變數中找不到 GOOGLE_API_KEY，請確認 Streamlit Secrets 設定。")
        st.stop()

    try:
        options = ClientOptions(api_key=GOOGLE_API_KEY)
        vision_client = vision.ImageAnnotatorClient(client_options=options)
        gemini_client = Client() 
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
            with st.spinner("正在執行精準座標對齊與文字提取..."):
                q_bytes = pdf_q.read()
                s_bytes = pdf_s.read()
                a_bytes = pdf_a.read()
                p_bytes = pdf_p.read()

                q_texts, _ = detect_and_ocr_boxes(q_bytes, vision_client)
                s_texts, s_debug_imgs = detect_and_ocr_boxes(s_bytes, vision_client, need_debug=True)
                a_texts, _ = detect_and_ocr_boxes(a_bytes, vision_client)
                p_texts, _ = detect_and_ocr_boxes(p_bytes, vision_client)
                
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
        
        # 讓使用者可以手動調整或即時查看的編輯器
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            height=600
        )
        # 確保編輯器的任何變動立刻同步回 session_state
        st.session_state.df = edited_df

        # --- 核心修正：每次渲染時，先即時計算當前 df 內的配分與得分 ---
        try:
            total_possible = pd.to_numeric(st.session_state.df["配分"]).sum()
            current_score = pd.to_numeric(st.session_state.df["得分"]).sum()
        except Exception:
            total_possible = 0
            current_score = 0

        # 建立按鈕與分數顯示欄位
        btn_col, score_col = st.columns([1, 1])

        with btn_col:
            run_ai = st.button("🤖 執行 Gemini 自動批改", use_container_width=True)
        
        with score_col:
            # 這裡會即時顯示加總成果
            st.markdown(f"### 🎯 目前得分：{current_score:.1f} / {total_possible:.1f}")

        # --- 執行 AI 批改邏輯 ---
        if run_ai:
            with st.spinner("Gemini Pro 正在深入分析答案並評分中..."):
                for index, row in st.session_state.df.iterrows():
                    if row["學生作答"]:
                        prompt = (
                            f"你是一位專業老師。請根據以下資訊評分：\n"
                            f"問題：{row['問題內容']}\n"
                            f"標準答案：{row['標準答案']}\n"
                            f"學生回答：{row['學生作答']}\n"
                            f"該題最高分（配分）：{row['配分']}\n"
                        )
                        try:
                            response = gemini_client.models.generate_content(
                                model='gemini-2.5-pro',
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=GradingResult,
                                    temperature=0.2, 
                                ),
                            )
                            
                            result: GradingResult = response.parsed
                            
                            # 寫入最新的得分與理由
                            st.session_state.df.at[index, "得分"] = float(result.score)
                            st.session_state.df.at[index, "AI 評分理由"] = result.reason
                            
                        except Exception as e:
                            st.warning(f"第 {index+1} 題評分錯誤: {e}")
                
                st.success("🎉 全數批改完成！")
                # 批改完成後強制全面刷新，讓最上方的 current_score 重新計算並完美顯示
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
