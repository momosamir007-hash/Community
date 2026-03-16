import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# إعدادات الصفحة
st.set_page_config(page_title="Sudoku AI Solver", layout="centered")

@st.cache_resource
def load_digit_model():
    try:
        return tf.keras.models.load_model('model.h5')
    except:
        return None

# دالة الحل الرياضي
def is_valid(b, r, c, n):
    if n in b[r,:] or n in b[:,c]: return False
    sr, sc = (r//3)*3, (c//3)*3
    return n not in b[sr:sr+3, sc:sc+3]

def solve(b):
    for r in range(9):
        for c in range(9):
            if b[r][c] == 0:
                for n in range(1, 10):
                    if is_valid(b, r, c, n):
                        b[r][c] = n
                        if solve(b): return True
                        b[r][c] = 0
                return False
    return True

# واجهة التطبيق
st.title("🧩 حل السودوكو: النتيجة النهائية")
model = load_digit_model()

if model is None:
    st.error("❌ ملف model.h5 غير موجود في المستودع!")
    st.stop()

file = st.file_uploader("ارفع صورة اللغز", type=['jpg','png','jpeg'])

if file:
    # قراءة الصورة
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # البحث عن الشبكة
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx
            break

    if pts is not None:
        # قص الصورة (Perspective Warp)
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
        
        M = cv2.getPerspectiveTransform(rect, np.float32([[0,0],[449,0],[449,449],[0,449]]))
        warped = cv2.warpPerspective(img, M, (450, 450))
        
        # استخراج الأرقام
        with st.spinner("🤖 جاري استخراج الأرقام..."):
            board = np.zeros((9, 9), dtype=int)
            w_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, w_thresh = cv2.threshold(w_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            for i in range(9):
                for j in range(9):
                    cell = w_thresh[i*50+5:i*50+45, j*50+5:j*50+45]
                    c_cnts, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if c_cnts:
                        c_max = max(c_cnts, key=cv2.contourArea)
                        if cv2.contourArea(c_max) > 30:
                            x,y,w,h = cv2.boundingRect(c_max)
                            digit = cell[y:y+h, x:x+w]
                            res = cv2.resize(digit, (18, 18))
                            canvas = np.zeros((28, 28))
                            canvas[5:23, 5:23] = res
                            pred = model.predict(canvas.reshape(1,28,28,1)/255.0, verbose=0)
                            board[i,j] = np.argmax(pred)

        st.write("📝 راجع الأرقام المكتشفة أدناه وصحح أي خطأ (0 يعني خلية فارغة):")
        board_df = st.data_editor(pd.DataFrame(board))

        if st.button("🚀 حل اللغز وملء الصورة"):
            final_board = board_df.to_numpy().astype(int)
            original_copy = final_board.copy()
            
            solved_board = final_board.copy()
            if solve(solved_board):
                # الرسم المباشر على النسخة المقصوصة
                result_img = warped.copy()
                for i in range(9):
                    for j in range(9):
                        if original_copy[i,j] == 0:
                            # وضع الرقم الأحمر في منتصف الخلية تماماً
                            cv2.putText(result_img, str(solved_board[i,j]), 
                                        (j*50+15, i*50+40), 
                                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 255), 3)
                
                st.success("🎯 تمت العملية! انظر للنتيجة أدناه:")
                st.image(result_img, channels="BGR", use_container_width=True)
            else:
                st.error("تعذر حل هذا اللغز رياضياً، تأكد من الأرقام في الجدول.")
    else:
        st.error("لم يتم العثور على شبكة سودوكو واضحة.")
