import streamlit as st
import cv2
import numpy as np
import pandas as pd
import fitz
import tensorflow as tf
import hashlib

# إعدادات الصفحة
st.set_page_config(page_title="Sudoku Solver AI", layout="centered")

@st.cache_resource
def load_digit_model():
    try:
        return tf.keras.models.load_model('model.h5')
    except:
        return None

# تهيئة الذاكرة
if 'board_extracted' not in st.session_state:
    st.session_state.update({'img_hash': None, 'board_extracted': False, 'extracted_board': None, 'solved_img': None})

# 1. المعالجة المسبقة
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# 2. البحث عن الشبكة
def find_sudoku(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

# 3. ترتيب النقاط
def order_pts(pts):
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

# 4. خوارزمية الحل
def is_valid(b, r, c, n):
    if n in b[r,:] or n in b[:,c]: return False
    sr, sc = (r//3)*3, (c//3)*3
    return n not in b[sr:sr+3, sc:sc+3]

def solve(b):
    for r in range(9):
        for c in range(9):
            if b[r,c] == 0:
                for n in range(1, 10):
                    if is_valid(b, r, c, n):
                        b[r,c] = n
                        if solve(b): return True
                        b[r,c] = 0
                return False
    return True

# 5. الرسم المباشر على الصورة (الحل الأكيد) 🔴
def draw_on_warped(warped_img, original_board, solved_board):
    result = warped_img.copy()
    h, w = result.shape[:2]
    ch, cw = h // 9, w // 9
    for i in range(9):
        for j in range(9):
            if original_board[i,j] == 0 and solved_board[i,j] != 0:
                text = str(solved_board[i,j])
                # رسم الرقم باللون الأحمر الفاقع مباشرة على الصورة المقصوصة
                cv2.putText(result, text, (j*cw+15, i*ch+35), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
    return result

# واجهة التطبيق
st.title("🧩 السودوكو الذكي: النتيجة المباشرة")
model = load_digit_model()

if model is None:
    st.error("❌ ارفع ملف model.h5 أولاً!")
    st.stop()

file = st.file_uploader("ارفع صورة السودوكو", type=['jpg','png','jpeg'])

if file:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    h_val = hashlib.md5(img.tobytes()[:5000]).hexdigest()
    
    if st.session_state.img_hash != h_val:
        st.session_state.update({'img_hash':h_val, 'board_extracted':False, 'solved_img':None})

    thresh = preprocess(img)
    pts = find_sudoku(thresh)

    if pts is not None:
        # قص الشبكة وتعديلها بمقاس ثابت 450x450
        rect = order_pts(pts)
        M = cv2.getPerspectiveTransform(rect, np.float32([[0,0],[449,0],[449,449],[0,449]]))
        warped = cv2.warpPerspective(img, M, (450, 450))
        
        if not st.session_state.board_extracted:
            with st.spinner("🤖 ذكاء اصطناعي..."):
                board = np.zeros((9,9), dtype=int)
                w_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, w_thresh = cv2.threshold(w_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                for i in range(9):
                    for j in range(9):
                        cell = w_thresh[i*50+5:i*50+45, j*50+5:j*50+45]
                        cnts, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            if cv2.contourArea(c) > 30:
                                x,y,w,h = cv2.boundingRect(c)
                                digit = cell[y:y+h, x:x+w]
                                canvas = np.zeros((28,28), dtype="float32")
                                res = cv2.resize(digit, (18,18))
                                canvas[5:23, 5:23] = res
                                pred = model.predict(canvas.reshape(1,28,28,1)/255.0, verbose=0)
                                if np.argmax(pred) != 0: board[i,j] = np.argmax(pred)
                
                st.session_state.extracted_board = board.copy()
                st.session_state.board_extracted = True

        # التحقق والحل
        if st.session_state.board_extracted:
            b = st.session_state.extracted_board.copy()
            st.write("📝 الأرقام المكتشفة (يمكنك التعديل إذا وجد خطأ):")
            df = st.data_editor(pd.DataFrame(b))
            
            if st.button("🚀 حل اللغز وملء الفراغات"):
                b_corrected = df.to_numpy().astype(int)
                s_board = b_corrected.copy()
                if solve(s_board):
                    # الرسم مباشرة على الصورة المقصوصة لضمان الظهور
                    st.session_state.solved_img = draw_on_warped(warped, b_corrected, s_board)
                    st.success("🎯 تم الحل! انظر إلى الصورة أدناه")
                else:
                    st.error("تعذر الحل رياضياً!")

        if st.session_state.solved_img is not None:
            st.image(st.session_state.solved_img, channels="BGR", use_container_width=True)
    else:
        st.error("لم يتم العثور على شبكة")

