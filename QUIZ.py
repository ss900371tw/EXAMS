import streamlit as st
import pandas as pd
import numpy as np
import cv2
import io
import os
import pdfplumber
from PIL import Image
from pdf2image import convert_from_bytes

# 引入最新的 Gemini GenAI SDK (負責視覺辨識與結構化批改)
from google.genai import Client as GeminiClient
from google.genai import types
from pydantic import BaseModel, Field

# --- Define Pydantic Schema for Gemini Structured Output ---

class GradingResult(BaseModel):
    score: float = Field(description="根據標準答案與學生作答圖片給予的得分，不可以超過該題的最高配分")
    reason: str = Field(description="詳細的評分理由與針對學生作答圖片的講評，若學生未作答或空白請說明")

# --- 核心影像處理與區域提取邏輯 ---

def order_points_robust(pts):
    """強健的點排序：左上, 右上, 右下, 左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # tl
    rect[2] = pts[np.argmax(s)]    # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def has_visible_content_in_crop(img_np, y_start, y_end, threshold_ratio=0.005):
    """針對掃描檔的盲區內容偵測"""
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

def detect_and_extract_blocks(pdf_bytes, min_w=400, min_h=100, return_images=False):
    """
    結合 OpenCV 定位。
    🚀 已優化：加入上下盲區安全過濾，防止頁碼、頁首干擾跨頁合併。
    """
    images = convert_from_bytes(pdf_bytes, dpi=350)
    all_results = [] 
    
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

            # =======================================================
            # 🚀 核心優化：消除頁底頁碼與頁首雜訊 (安全邊緣遮罩)
            # 考卷底部正中間的頁碼通常落在底部 5% ~ 6% 的區間，我們將其在二值化前排除
            # =======================================================
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            top_margin = int(h_img * 0.05)     # 頂部 5% 視為頁首盲區
            bottom_margin = int(h_img * 0.94)  # 底部 6% 視為頁尾頁碼盲區
            
            # 建立一個乾淨的二值化參考圖，將頁首頁尾直接填白（即不偵測任何線條與文字）
            analysis_gray = gray.copy()
            analysis_gray[0:top_margin, :] = 255
            analysis_gray[bottom_margin:h_img, :] = 255
            # =======================================================

            blurred = cv2.GaussianBlur(analysis_gray, (5, 5), 0)
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
                
                extracted_text = ""
                try:
                    crop = page_plumber.within_bbox(pdf_bbox)
                    extracted_text = (crop.extract_text() or "").strip()
                except Exception:
                    extracted_text = ""

                cropped_img = None
                rect_pts = order_points_robust(box_points)
                dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
                warped = cv2.warpPerspective(img_np, M, (int(w), int(h)))
                
                pad = 10
                if w > pad*2 and h > pad*2:
                    warped = warped[pad:int(h)-pad, pad:int(w)-pad]
                
                cropped_img = Image.fromarray(warped)

                page_blocks[p_idx].append({
                    "text": extracted_text,
                    "image": cropped_img,
                    "bbox": pdf_bbox,
                    "img_box": (x_min, y_min, x_max, y_max)
                })

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
                    # 🚀 優化：在這裡同樣限縮檢查範圍，避開底部 6% 的頁碼純文字區
                    bottom_crop = prev_page_plumber.crop((0, last_block_prev_page["bbox"][3], prev_page_plumber.width, prev_page_plumber.height * 0.94))
                    text_below_last_table = bottom_crop.extract_text() or ""
                    
                    top_crop = curr_page_plumber.crop((0, curr_page_plumber.height * 0.05, curr_page_plumber.width, first_block_curr_page["bbox"][0]))
                    text_above_first_table = top_crop.extract_text() or ""
                except Exception:
                    pass
                
                has_digital_text_between = bool(text_below_last_table.strip() or text_above_first_table.strip())
                
                prev_img_np = page_cv_images[p_idx - 1]
                curr_img_np = page_cv_images[p_idx]
                
                _, _, _, last_y_max = last_block_prev_page["img_box"]
                _, first_y_min, _, _ = first_block_curr_page["img_box"]
                
                # 🚀 優化：盲區偵測時，限制終點在 bottom_margin 之前，起點在 top_margin 之後
                has_scanned_content_below = has_visible_content_in_crop(prev_img_np, last_y_max, int(prev_img_np.shape[0] * 0.94))
                has_scanned_content_above = has_visible_content_in_crop(curr_img_np, int(curr_img_np.shape[0] * 0.05), first_y_min)
                
                has_image_content_between = has_scanned_content_below or has_scanned_content_above
                
                if (not has_digital_text_between) and (not has_image_content_between):
                    last_block_prev_page["text"] += "\n" + first_block_curr_page["text"]
                    
                    img1 = last_block_prev_page["image"]
                    img2 = first_block_curr_page["image"]
                    dst = Image.new('RGB', (max(img1.width, img2.width), img1.height + img2.height))
                    dst.paste(img1, (0, 0))
                    dst.paste(img2, (0, img1.height))
                    last_block_prev_page["image"] = dst
                    
                    first_block_curr_page["is_merged_child"] = True

        for p_idx in range(total_pages):
            for block in page_blocks[p_idx]:
                if not block.get("is_merged_child", False):
                    if return_images:
                        all_results.append(block["image"]) 
                    else:
                        all_results.append(block["text"])  

    return all_results

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 多模態評分系統", layout="wide")
    st.title("📑 全自動考卷批改工作台 (Gemini 2.0 原生多模態免 OCR 版)")

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["題目", "問題內容", "學生作答(影像物件)", "標準答案", "配分", "得分", "AI 評分理由"])

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("❌ 環境變數中找不到 GOOGLE_API_KEY，請確認 Streamlit Secrets 設定。")
        st.stop()

    try:
        # 初始化 Gemini 新版 Client 
        gemini_client = GeminiClient() 
    except Exception as e:
        st.error(f"💥 初始化 Gemini API 失敗：{e}")
        return

    with st.sidebar:
        st.header("📥 上傳 PDF 來源")
        pdf_q = st.file_uploader("1. 問題內容 PDF", type="pdf")
        pdf_s = st.file_uploader("2. 學生作答 PDF (支援手寫/影像)", type="pdf")
        pdf_a = st.file_uploader("3. 標準答案 PDF", type="pdf")
        pdf_p = st.file_uploader("4. 配分 PDF", type="pdf")
        
        if st.button("🚀 開始全自動解析") and pdf_q and pdf_s and pdf_a and pdf_p:
            with st.spinner("正在執行結構化座標對齊與多模態區塊裁切..."):
                q_bytes = pdf_q.read()
                s_bytes = pdf_s.read()
                a_bytes = pdf_a.read()
                p_bytes = pdf_p.read()

                # 問題、答案、配分提取文字
                q_texts = detect_and_extract_blocks(q_bytes, return_images=False)
                a_texts = detect_and_extract_blocks(a_bytes, return_images=False)
                p_texts = detect_and_extract_blocks(p_bytes, return_images=False)
                
                # 💡 學生作答直接提取「影像物件列表」
                s_images = detect_and_extract_blocks(s_bytes, return_images=True)
                
                num_questions = max(len(q_texts), len(s_images), len(a_texts), len(p_texts))
                
                data = []
                for i in range(num_questions):
                    data.append({
                        "題目": f"第 {i+1} 題",
                        "問題內容": q_texts[i] if i < len(q_texts) else "",
                        "學生作答(影像物件)": s_images[i] if i < len(s_images) else None, # 儲存 PIL.Image
                        "標準答案": a_texts[i] if i < len(a_texts) else "",
                        "配分": p_texts[i] if i < len(p_texts) else "10",
                        "得分": 0.0,
                        "AI 評分理由": ""
                    })
                
                st.session_state.df = pd.DataFrame(data)
                st.success(f"解析完成！共對齊 {num_questions} 個多模態區塊。")

# --- 畫面佈局與舊版相容連動邏輯 ---
    col1, col2 = st.columns([7, 5])
    
    with col1:
        st.subheader("📝 結構化評分表")
        
        display_df = st.session_state.df.copy()
        display_df["學生作答(影像物件)"] = display_df["學生作答(影像物件)"].apply(
            lambda x: "📷 影像已就緒 (請用下方下拉選單切換檢視)" if x is not None else "⚠️ 無影像"
        )

        # 💡 移除舊版本不支援的 on_select 與 selection_mode
        edited_display_df = st.data_editor(
            display_df,
            num_rows="dynamic",
            use_container_width=True,
            height=400,
            key="my_data_editor"
        )
        
        # 同步資料
        if len(edited_display_df) == len(st.session_state.df):
            st.session_state.df["得分"] = edited_display_df["得分"]
            st.session_state.df["AI 評分理由"] = edited_display_df["AI 評分理由"]
            st.session_state.df["問題內容"] = edited_display_df["問題內容"]
            st.session_state.df["標準答案"] = edited_display_df["標準答案"]
            st.session_state.df["配分"] = edited_display_df["配分"]

        try:
            current_score = pd.to_numeric(st.session_state.df["得分"]).sum()
        except Exception:
            current_score = 0

        btn_col, score_col = st.columns([1, 1])
        with btn_col:
            run_ai = st.button("🤖 執行 Gemini 多模態自動批改", use_container_width=True)
        with score_col:
            st.markdown(f"### 🎯 目前總分：{current_score:.1f} / 100.0")

        # 🌟【舊版相容的連動神器】🌟 在表格下方放一個下拉選單供切換題號
        st.write("---")
        st.subheader("🔍 切換檢視題號")
        if len(st.session_state.df) > 0:
            q_list = st.session_state.df["題目"].tolist()
            selected_q_name = st.selectbox("請選擇你想在右側複核的題目：", q_list, index=0)
            # 找出選中題目的 index
            selected_idx = q_list.index(selected_q_name)
        else:
            selected_idx = None

        # --- 執行 多模態 AI 批改邏輯 ---
        if run_ai:
            temp_df = st.session_state.df.copy()
            num_rows = len(temp_df)
            status_text = st.empty()
            
            for index, row in temp_df.iterrows():
                status_text.markdown(f"⏳ **Gemini 正在視覺辨識並批改第 {index + 1} / {num_rows} 題...**")
                student_img = row["學生作答(影像物件)"]
                
                if student_img is not None:
                    prompt = (
                        f"你是一位溫和、具鼓勵性質的專業審查老師。目前正在批改學生的作答內容，評分核心原則為「從寬給分」，旨在確認學生是否掌握核心觀念，並鼓勵其學習動機。\n\n"
                        f"【單題題目資訊】\n"
                        f"問題內容：{row['問題內容']}\n"
                        f"標準答案：{row['標準答案']}\n"
                        f"最高配分（滿分）：{row['配分']} 分。\n\n"
                        f"【從寬給分（應變調整）指引】\n"
                        f"1. 觀念正確即給分：只要學生手寫內容、算式或圖表中展現出「核心觀念正確」，即使漏掉部分關鍵字、描述不夠精準或有輕微字詞筆誤，皆應盡量給予滿分或接近滿分。\n"
                        f"2. 同義轉換認可：不盲目進行字面精確對齊。若學生使用的術語、換句話說的表達方式與標準答案意思相近，視為正確。\n"
                        f"3. 局部給分（Partial Credit）從優：若算式推導過程正確但最後計算粗心出錯，或多步驟問題中前半段邏輯正確，部應該視為整題全錯。\n"
                        f"4. 錯誤處置：只有在完全答非所問、空白、或核心觀念嚴重錯誤時，才予以扣除較多分數。\n\n"
                        f"【批改任務說明】\n"
                        f"1. 請仔細審視附加的圖片，圖片內包含學生針對該題的手寫答案、算式、圖表或打字內容。\n"
                        f"2. 遵循「從寬給分」指引，對照標準答案，給予 0 到 {row['配分']} 之間的合理分數。\n"
                        f"3. 評分理由請詳細列出：\n"
                        f"   - 學生在哪個核心觀念上表現正確（讚賞點）。\n"
                        f"   - 若有扣分，請具體且溫和地指出圖片中推導的瑕疵或漏掉的關鍵部分，並說明此分數是如何「從寬考量」後給出的。\n"
                        f"4. 嚴重警告：你給出的分數「絕對不可以」超過該題的最高配分。\n"
                    )
                    try:
                        response = gemini_client.models.generate_content(
                            model='gemini-2.5-pro',  
                            contents=[prompt, student_img],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=GradingResult,
                                temperature=0.1, 
                            ),
                        )
                        result: GradingResult = response.parsed
                        temp_df.at[index, "得分"] = float(result.score)
                        temp_df.at[index, "AI 評分理由"] = result.reason
                    except Exception as e:
                        st.warning(f"第 {index+1} 題多模態評分發生錯誤: {e}")
                else:
                    temp_df.at[index, "得分"] = 0.0
                    temp_df.at[index, "AI 評分理由"] = "未偵測到學生作答圖片，以 0 分計算。"
            
            status_text.empty()
            st.session_state.df = temp_df
            st.success("🎉 全數多模態批改完成！")
            st.rerun()
            
    with col2:
        st.subheader("🔍 盲區視覺複核面板")
        
        if selected_idx is not None and selected_idx < len(st.session_state.df):
            row_data = st.session_state.df.iloc[selected_idx]
            st.markdown(f"#### 📋 當前檢視：**{row_data['題目']}**")
            st.markdown(f"**問題：** {row_data['問題內容']}")
            st.markdown(f"**標準答案：** {row_data['標準答案']}")
            
            img_obj = row_data["學生作答(影像物件)"]
            if img_obj is not None:
                st.image(img_obj, use_container_width=True, caption=f"第 {selected_idx+1} 題 學生作答盲區裁剪影像")
            else:
                st.warning("⚠️ 該題無對應的學生作答影像")
        else:
            st.info("💡 項目載入後，右側將即時顯示指定題目的原始手寫作答圖片。")

if __name__ == "__main__":
    main()

