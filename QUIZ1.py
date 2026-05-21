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


def detect_and_ocr_boxes(pdf_bytes, vision_client, min_w=200, min_h=100, need_debug=False, external_boxes=None):
    """
    結合 OpenCV 定位與 pdfplumber/Google Vision 的混合文字提取。
    優先順序：pdfplumber 區域提取 > Google Vision OCR。
    
    支援 external_boxes：若傳入外部 Bounding Boxes 結構，則跳過框線偵測，直接沿用座標。
    """
    images = convert_from_bytes(pdf_bytes, dpi=350)
    all_text_results = []
    debug_images = []
    all_page_candidates = [] # 用來記錄每一頁的 Box 資訊

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p_idx, img in enumerate(images):
            img_np = np.array(img.convert("RGB"))
            h_img, w_img, _ = img_np.shape
            
            page_plumber = pdf.pages[p_idx]
            w_pdf, h_pdf = page_plumber.width, page_plumber.height
            
            scale_x = w_pdf / w_img
            scale_y = h_pdf / h_img

            # 💡 核心改動：如果外部有提供對齊用的 Box 座標，直接借用
            if external_boxes is not None and p_idx < len(external_boxes):
                candidates = external_boxes[p_idx]
            else:
                # 執行原本的 OpenCV 框線偵測邏輯
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
                        box = np.int0(box)
                        candidates.append(((cx, cy), (w, h), angle, box))

                candidates = sorted(candidates, key=lambda x: x[0][1])

            # 儲存這一頁最終使用的 candidates 座標，供後續回傳
            all_page_candidates.append(candidates)

            # 開始依據座標切圖與提取文字
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

                all_text_results.append(text)
                
                if need_debug:
                    cv2.drawContours(img_np, [box_points], 0, (0, 255, 0), 3)

            if need_debug:
                debug_images.append(Image.fromarray(img_np))

    return all_text_results, debug_images, all_page_candidates

# --- 2. Streamlit UI ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    st.set_page_config(page_title="AI 評分系統", layout="wide")
    st.title("📑 全自動考卷批改工作台 (Gemini Pro 強化版)")

    if "debug_images" not in st.session_state:
        st.session_state.debug_images = []

    try:
        vision_client = vision.ImageAnnotatorClient()
        gemini_client = Client() 
    except Exception as e:
        st.error(f"初始化 API 失敗，請檢查環境變數與憑證：{e}")
        return

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["題目", "問題內容", "學生作答", "標準答案", "配分", "得分", "AI 評分理由"])

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

                # 💡 調整回傳值接收：現在函式會多回傳一個 Box 座標清單
                q_texts, _, _ = detect_and_ocr_boxes(q_bytes, vision_client)
                a_texts, _, _ = detect_and_ocr_boxes(a_bytes, vision_client)
                p_texts, _, p_boxes = detect_and_ocr_boxes(p_bytes, vision_client)
                
                # 嘗試正常偵測學生作答的 Box
                s_texts, s_debug_imgs, s_boxes = detect_and_ocr_boxes(s_bytes, vision_client, need_debug=True)
                
                # 💡 核心對齊邏輯：計算學生 PDF 偵測到的總 Box 數量
                total_s_boxes = sum(len(page_boxes) for page_boxes in s_boxes)
                
                # 如果學生 PDF 沒抓到半個 Box，自動套用配分 PDF 的座標重新提取
                if total_s_boxes == 0:
                    st.sidebar.info("🔍 偵測到學生作答未包含明晰框線，已啟動自動對齊：正在套用「配分 PDF」的區域座標進行強制 OCR...")
                    s_texts, s_debug_imgs, _ = detect_and_ocr_boxes(
                        s_bytes, vision_client, need_debug=True, external_boxes=p_boxes
                    )
                
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
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            height=600
        )
        st.session_state.df = edited_df

        try:
            total_possible = pd.to_numeric(st.session_state.df["配分"]).sum()
            current_score = pd.to_numeric(st.session_state.df["得分"]).sum()
        except:
            total_possible = 0
            current_score = 0

        btn_col, score_col = st.columns([1, 1])

        with btn_col:
            run_ai = st.button("🤖 執行 Gemini 自動批改", use_container_width=True)
        
        with score_col:
            st.markdown(f"### 🎯 目前得分：{current_score:.1f} / {total_possible:.1f}")

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
                            st.session_state.df.at[index, "得分"] = result.score
                            st.session_state.df.at[index, "AI 評分理由"] = result.reason
                            
                        except Exception as e:
                            st.warning(f"第 {index+1} 題評分錯誤: {e}")
                st.rerun()

    with col2:
        st.subheader("🔍 框選對齊視覺檢查")
        # 💡 修正原本判斷條件寫錯的 Bug (原本寫 debug_imgs 導致綠框預覽出不來)
        if "debug_images" in st.session_state:
            for i, img in enumerate(st.session_state.debug_images):
                st.image(img, caption=f"偵測區域預覽 - 頁面 {i+1}")

if __name__ == "__main__":
    main()
