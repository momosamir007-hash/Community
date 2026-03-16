import streamlit as st
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import time
import tensorflow as tf
import hashlib

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Sudoku AI Master Solver",
    page_icon="🤖",
    layout="centered"
)

# تهيئة الذاكرة (Session State)
for key, default in {
    'img_hash': None,
    'board_extracted': False,
    'extracted_board': None,
    'solved_img': None,
    'solved_board': None,
    'debug_clean': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 0. تحميل النموذج الذكي
# ==========================================
@st.cache_resource
def load_digit_model():
    model_path = 'model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        return None

# ==========================================
# 1. دوال معالجة الصور الأساسية
# ==========================================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

def find_board(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                best = approx
    return best

def order_points(pts):
    pts = pts.reshape((4, 2)).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect

def warp_image(img, pts, size=450):
    src = order_points(pts)
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size)), M

# ==========================================
# 2. استخراج الأرقام بالذكاء الاصطناعي 🧠
# ==========================================
def smart_extract_digits_cnn(warped_img, model):
    board = np.zeros((9, 9), dtype=int)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    debug_montage = np.zeros((9*28, 9*28), dtype=np.uint8)
    progress = st.progress(0, text="🤖 الذكاء الاصطناعي يحلل الأرقام...")
    
    cell_size = 450 // 9
    for i in range(9):
        for j in range(9):
            # قص الخلية مع هامش ذكي
            m = int(cell_size * 0.12)
            cell = thresh[i*cell_size+m : (i+1)*cell_size-m, j*cell_size+m : (j+1)*cell_size-m]
            
            contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 30:
                    x, y, w, h = cv2.boundingRect(c)
                    digit = cell[y:y+h, x:x+w]
                    
                    # تجهيز للمقاس 28x28 (MNIST Style)
                    canvas = np.zeros((28, 28), dtype=np.uint8)
                    scale = 20.0 / max(w, h)
                    nw, nh = int(w*scale), int(h*scale)
                    res = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)
                    canvas[(28-nh)//2 : (28-nh)//2+nh, (28-nw)//2 : (28-nw)//2+nw] = res
                    
                    debug_montage[i*28:(i+1)*28, j*28:(j+1)*28] = canvas
                    
                    # التنبؤ
                    inp = canvas.reshape(1, 28, 28, 1).astype('float32') / 255.0
                    pred = model.predict(inp, verbose=0)
                    if np.max(pred) > 0.7 and np.argmax(pred) != 0:
                        board[i][j] = np.argmax(pred)
            
            progress.progress(((i * 9 + j) + 1) / 81)
            
    progress.empty()
    st.session_state.debug_clean = debug_montage
    return board

# ==========================================
# 3. محرك حل السودوكو
# ==========================================
def is_valid(b, r, c, n):
    if n in b[r, :] or n in b[:, c]: return False
    sr, sc = (r//3)*3, (c//3)*3
    if n in b[sr:sr+3, sc:sc+3]: return False
    return True

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

def validate_board(board):
    temp = board.copy()
    for i in range(9):
        for j in range(9):
            v = temp[i,j]
            if v != 0:
                temp[i,j] = 0
                if not is_valid(temp, i, j, v): return False, f"تكرار الرقم {v}"
                temp[i,j] = v
    return True, ""

# ==========================================
# 4. رسم الحل (باللون الأحمر البارز) 🔴
# ==========================================
def draw_solution(img, solved, original):
    h, w = img.shape[:2]
    ch, cw = h // 9, w // 9
    for i in range(9):
        for j in range(9):
            if original[i,j] == 0 and solved[i,j] != 0:
                txt = str(solved[i,j])
                font = cv2.FONT_HERSHEY_DUPLEX
                fs, thick = 1.3, 3
                size = cv2.getTextSize(txt, font, fs, thick)[0]
                tx = j*cw + (cw - size[0])//2
                ty = i*ch + (ch + size[1])//2
                # رسم الرقم باللون الأحمر
                cv2.putText(img, txt, (tx, ty), font, fs, (0, 0, 255), thick)
    return img

# ==========================================
# واجهة التطبيق
# ==========================================
st.title("🤖 حلّال السودوكو الذكي النهائي")
st.caption("يستخدم نموذج CNN مخصص لرؤية الأرقام وحلها آلياً")

model = load_digit_model()
if model is None:
    st.error("⚠️ ملف model.h5 مفقود!")
    st.stop()

choice = st.radio("المصدر:", ("📸 كاميرا", "📁 ملف"), horizontal=True)
upload = st.camera_input("صوّر اللغز") if choice == "📸 كاميرا" else st.file_uploader("ارفع صورة", type=['png','jpg','jpeg','pdf'])

if upload:
    data = upload.getvalue()
    if upload.name.lower().endswith('.pdf'):
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=200)
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), 1)
        doc.close()
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

    if img is not None:
        h_val = hashlib.md5(img.tobytes()[:5000]).hexdigest()
        if st.session_state.img_hash != h_val:
            st.session_state.update({'img_hash':h_val, 'board_extracted':False, 'extracted_board':None, 'solved_img':None})

        thresh = preprocess_image(img)
        pts = find_board(thresh)

        if pts is not None:
            warped, M = warp_image(img, pts)
            
            if not st.session_state.board_extracted:
                board = smart_extract_digits_cnn(warped, model)
                if board is not None:
                    st.session_state.update({'extracted_board':board, 'board_extracted':True})

            if st.session_state.board_extracted and st.session_state.solved_img is None:
                b = st.session_state.extracted_board.copy()
                ok, msg = validate_board(b)
                
                if not ok:
                    st.warning(f"⚠️ {msg}. صحح الأرقام أدناه:")
                    df = pd.DataFrame(b, columns=[f"C{i+1}" for i in range(9)])
                    new_df = st.data_editor(df)
                    if st.button("🚀 حل الآن"):
                        b = new_df.to_numpy().astype(int)
                        ok2, _ = validate_board(b)
                        if ok2:
                            s_board = b.copy()
                            if solve(s_board):
                                ar = draw_solution(np.zeros((450,450,3), np.uint8), s_board, b)
                                inv_M = cv2.getPerspectiveTransform(np.float32([[0,0],[449,0],[449,449],[0,449]]), order_points(pts))
                                inv_w = cv2.warpPerspective(ar, inv_M, (img.shape[1], img.shape[0]))
                                st.session_state.solved_img = cv2.addWeighted(img, 1, inv_w, 1, 0)
                                st.rerun()
                else:
                    s_board = b.copy()
                    if solve(s_board):
                        ar = draw_solution(np.zeros((450,450,3), np.uint8), s_board, b)
                        inv_M = cv2.getPerspectiveTransform(np.float32([[0,0],[449,0],[449,449],[0,449]]), order_points(pts))
                        inv_w = cv2.warpPerspective(ar, inv_M, (img.shape[1], img.shape[0]))
                        st.session_state.solved_img = cv2.addWeighted(img, 1, inv_w, 1, 0)
                        st.success("✅ تم الحل بنجاح!")
                        st.balloons()

            if st.session_state.solved_img is not None:
                st.image(st.session_state.solved_img, channels="BGR", use_container_width=True)
                with st.expander("🛠️ المعاينة التقنية"):
                    st.image(st.session_state.debug_clean, caption="الأرقام كما رآها الذكاء الاصطناعي")
        else:
            st.error("❌ لم يتم العثور على شبكة.")
