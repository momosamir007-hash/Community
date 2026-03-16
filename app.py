import streamlit as st
import cv2
import numpy as np
import pandas as pd
import fitz
import time
import tensorflow as tf

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Sudoku AR Solver (AI CNN)",
    page_icon="🤖",
    layout="centered"
)

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
# 0. تحميل نموذج الذكاء الاصطناعي
# ==========================================
@st.cache_resource
def load_digit_model():
    model_path = 'model.h5' 
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None

# ==========================================
# 1. دوال معالجة الصور
# ==========================================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh

def find_board(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 20000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best = approx
    return best

def order_points(pts):
    pts = pts.reshape((4, 2)).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def warp_image(img, pts, size=450):
    src = order_points(pts)
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, M

# ==========================================
# 2. الاستخراج والتجهيز للذكاء الاصطناعي
# ==========================================
def get_cell_inner(thresh_img, row, col, grid_size=450, margin=0.15):
    cell_size = grid_size // 9
    y1 = row * cell_size
    x1 = col * cell_size
    cell = thresh_img[y1:y1 + cell_size, x1:x1 + cell_size]
    m = int(cell_size * margin)
    inner = cell[m:cell_size - m, m:cell_size - m]
    return inner

def smart_extract_digits_cnn(warped_img, model):
    board = np.zeros((9, 9), dtype=int)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    debug_montage = np.zeros((9*28, 9*28), dtype=np.uint8) 
    progress = st.progress(0, text="🤖 الذكاء الاصطناعي يقرأ الشبكة الآن...")
    
    cells_processed = 0
    for i in range(9):
        for j in range(9):
            cell_inner = get_cell_inner(thresh, i, j, margin=0.10) 
            kernel = np.ones((2,2), np.uint8)
            cell_inner = cv2.morphologyEx(cell_inner, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(cell_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 25: 
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = h / float(w)
                    if 0.2 < aspect_ratio < 5.0:
                        digit_roi = cell_inner[y:y+h, x:x+w]
                        canvas = np.zeros((28, 28), dtype=np.uint8)
                        scale = 20.0 / max(w, h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                        start_x = (28 - new_w) // 2
                        start_y = (28 - new_h) // 2
                        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit
                        
                        # إزالة فلتر التسميك (dilate) لأنه قد يغلق فتحة الرقم 5 ليجعله 6
                        debug_montage[i*28:(i+1)*28, j*28:(j+1)*28] = canvas
                        
                        cnn_input = canvas.reshape(1, 28, 28, 1).astype('float32') / 255.0
                        prediction = model.predict(cnn_input, verbose=0)
                        
                        predicted_digit = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        if confidence > 0.60 and predicted_digit != 0:
                            board[i][j] = predicted_digit
                            
            cells_processed += 1
            progress.progress(cells_processed / 81)
            
    progress.empty()
    st.session_state.debug_clean = debug_montage
    return board

# ==========================================
# 3. خوارزمية الحل والتحقق
# ==========================================
def is_valid(board, r, c, num):
    for i in range(9):
        if board[r][i] == num or board[i][c] == num:
            return False
    sr, sc = r - r % 3, c - c % 3
    for i in range(3):
        for j in range(3):
            if board[sr + i][sc + j] == num:
                return False
    return True

def solve_sudoku(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for n in range(1, 10):
                    if is_valid(board, r, c, n):
                        board[r][c] = n
                        if solve_sudoku(board):
                            return True
                        board[r][c] = 0
                return False
    return True

def validate_board(board):
    temp = board.copy()
    for i in range(9):
        for j in range(9):
            v = temp[i][j]
            if v < 0 or v > 9:
                return False, f"قيمة خارج النطاق ({v}) في الصف {i+1} والعمود {j+1}"
            if v != 0:
                temp[i][j] = 0
                if not is_valid(temp, i, j, v):
                    temp[i][j] = v
                    return False, f"تعارض: الرقم {v} متكرر في نفس الصف أو العمود أو المربع."
                temp[i][j] = v
    return True, ""

# ==========================================
# 4. الواقع المعزز
# ==========================================
def draw_solution(img, solved, original):
    ch = img.shape[0] // 9
    cw = img.shape[1] // 9
    for i in range(9):
        for j in range(9):
            if original[i][j] == 0 and solved[i][j] != 0:
                text = str(solved[i][j])
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                tx = j * cw + (cw - ts[0]) // 2
                ty = i * ch + (ch + ts[1]) // 2
                cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return img

# ==========================================
# واجهة المستخدم
# ==========================================
st.title("🤖 حلّال السودوكو الذكي (مع المصحح الآلي)")
st.write("التقط صورة وسنقوم بالحل. إذا أخطأ الذكاء الاصطناعي في رقم، يمكنك تصحيحه بسهولة!")

cnn_model = load_digit_model()
if cnn_model is None:
    st.error("⚠️ لم يتم العثور على ملف `model.h5`.")
    st.stop()

method = st.radio("طريقة الإدخال:", ("📸 كاميرا", "📁 ملف (صورة/PDF)"), horizontal=True)

cv2_img = None

if method == "📸 كاميرا":
    cam = st.camera_input("التقط صورة للسودوكو")
    if cam:
        cv2_img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
elif method == "📁 ملف (صورة/PDF)":
    file = st.file_uploader("ارفع صورة أو PDF", type=['png', 'jpg', 'jpeg', 'pdf'])
    if file:
        if file.name.lower().endswith('.pdf'):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            pix = doc[0].get_pixmap(dpi=200)
            cv2_img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
            doc.close()
        else:
            cv2_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

if cv2_img is not None:
    img_hash = hash(cv2_img.tobytes()[:8000])
    if st.session_state.img_hash != img_hash:
        st.session_state.img_hash = img_hash
        st.session_state.board_extracted = False
        st.session_state.extracted_board = None
        st.session_state.solved_img = None

    thresh = preprocess_image(cv2_img)
    contour = find_board(thresh)

    if contour is not None:
        SIZE = 450
        warped, warp_matrix = warp_image(cv2_img, contour, SIZE)

        # 1. الاستخراج
        if not st.session_state.board_extracted:
            with st.spinner("⏳ جاري تحليل الصورة..."):
                board = smart_extract_digits_cnn(warped, cnn_model)
                if board is not None and np.count_nonzero(board) > 0:
                    st.session_state.extracted_board = board.copy()
                    st.session_state.board_extracted = True
                else:
                    st.error("❌ لم نتمكن من العثور على أرقام واضحة.")

        # 2. التحقق والتصحيح
        if st.session_state.board_extracted and st.session_state.solved_img is None:
            current_board = st.session_state.extracted_board.copy()
            ok, msg = validate_board(current_board)
            
            if not ok:
                st.warning(f"⚠️ {msg}")
                st.info("الذكاء الاصطناعي أخطأ في قراءة رقم (مثل قراءة 5 على أنه 6). يرجى تصحيح الخطأ في الجدول أدناه (0 = خلية فارغة):")
                
                # عرض جدول قابل للتعديل
                df = pd.DataFrame(current_board, columns=[f"C{i}" for i in range(1, 10)])
                edited_df = st.data_editor(df, use_container_width=True)
                
                if st.button("🚀 أكمل الحل بعد التصحيح", type="primary"):
                    corrected_board = edited_df.to_numpy().astype(int)
                    ok2, msg2 = validate_board(corrected_board)
                    if ok2:
                        solved = corrected_board.copy()
                        if solve_sudoku(solved):
                            st.session_state.extracted_board = corrected_board # تحديث الأساس
                            
                            ar = np.zeros((SIZE, SIZE, 3), np.uint8)
                            ar = draw_solution(ar, solved, corrected_board)
                            inv_M = cv2.getPerspectiveTransform(
                                np.float32([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]]), order_points(contour)
                            )
                            inv_warp = cv2.warpPerspective(ar, inv_M, (cv2_img.shape[1], cv2_img.shape[0]))
                            st.session_state.solved_img = cv2.addWeighted(cv2_img, 1, inv_warp, 1, 0)
                            st.rerun() # تحديث الصفحة لعرض النتيجة
                        else:
                            st.error("⚠️ لا يزال اللغز غير قابل للحل رياضياً.")
                    else:
                        st.error(f"❌ التعديل غير صحيح: {msg2}")
            else:
                # إذا كانت صحيحة 100%، يحلها فوراً
                solved = current_board.copy()
                if solve_sudoku(solved):
                    ar = np.zeros((SIZE, SIZE, 3), np.uint8)
                    ar = draw_solution(ar, solved, current_board)
                    inv_M = cv2.getPerspectiveTransform(
                        np.float32([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]]), order_points(contour)
                    )
                    inv_warp = cv2.warpPerspective(ar, inv_M, (cv2_img.shape[1], cv2_img.shape[0]))
                    st.session_state.solved_img = cv2.addWeighted(cv2_img, 1, inv_warp, 1, 0)
                    st.success("✅ تم التعرف والحل آلياً بنسبة 100%!")
                    st.balloons()
                else:
                    st.error("⚠️ اللغز المستخرج غير قابل للحل.")

        # 3. عرض النتيجة النهائية
        if st.session_state.solved_img is not None:
            st.image(st.session_state.solved_img, channels="BGR", caption="✨ الحل النهائي (الواقع المعزز)", use_container_width=True)
            
        if st.session_state.debug_clean is not None:
            with st.expander("🛠️ كيف رأى الذكاء الاصطناعي الأرقام؟"):
                st.image(st.session_state.debug_clean, use_container_width=True)

    else:
        st.error("❌ لم يتم العثور على شبكة سودوكو واضحة.")
