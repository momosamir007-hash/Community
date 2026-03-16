import streamlit as st
import cv2
import numpy as np
import pandas as pd
import fitz
import time
import pytesseract

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Sudoku AR Solver (Tesseract AI)",
    page_icon="🧩",
    layout="centered"
)

# ==========================================
# تهيئة session_state
# ==========================================
for key, default in {
    'img_hash': None,
    'board_extracted': False,
    'original_board': None,
    'solved_img': None,
    'solved_board': None,
    'debug_clean': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 1. دوال معالجة الصور (تقنيات التحسين)
# ==========================================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين
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
# 2. الاستخراج الدقيق
# ==========================================
def get_cell_inner(gray, row, col, grid_size=450, margin=0.15):
    cell_size = grid_size // 9
    y1 = row * cell_size
    x1 = col * cell_size
    cell = gray[y1:y1 + cell_size, x1:x1 + cell_size]
    # قص الهوامش بقوة للتخلص من خطوط الشبكة
    m = int(cell_size * margin)
    inner = cell[m:cell_size - m, m:cell_size - m]
    return inner

def is_cell_filled(cell_gray):
    _, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    total = thresh.shape[0] * thresh.shape[1]
    if total == 0: return False
    white = cv2.countNonZero(thresh)
    ratio = white / total
    if ratio < 0.02 or ratio > 0.85:
        return False
    return True

# ==========================================
# 3. استخراج الأرقام عبر Tesseract محلياً 🌟
# ==========================================
def smart_extract_digits(warped_img):
    board = np.zeros((9, 9), dtype=int)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    
    debug_montage = np.ones((9*100, 9*100), dtype=np.uint8) * 255 # خلفية بيضاء
    
    cells_to_process = []
    for i in range(9):
        for j in range(9):
            inner_normal = get_cell_inner(gray, i, j)
            if is_cell_filled(inner_normal):
                cells_to_process.append((i, j, inner_normal))
                
    if not cells_to_process:
        return None
        
    total = len(cells_to_process)
    progress = st.progress(0, text=f"جاري قراءة {total} رقم باستخدام Tesseract...")
    
    # إعدادات Tesseract الصارمة: رقم واحد فقط، ومن 1 إلى 9
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
    
    for idx, (i, j, cell_img) in enumerate(cells_to_process):
        # تحويل الخلية لأبيض وأسود عالي التباين
        _, thresh = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # البحث عن الرقم داخل الخلية
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 20: # تجاهل النقط الصغيرة
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # قص الرقم فقط
                digit_roi = thresh[y:y+h, x:x+w]
                
                # إنشاء لوحة بيضاء 100x100
                canvas = np.ones((100, 100), dtype=np.uint8) * 255
                
                # تكبير الرقم
                scale = 60.0 / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # جعل الرقم أسود (لأن Tesseract يفضل الأسود على الأبيض)
                resized_digit = cv2.bitwise_not(resized_digit)
                
                # توسيط الرقم في اللوحة البيضاء
                start_x = (100 - new_w) // 2
                start_y = (100 - new_h) // 2
                canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit
                
                # وضع اللوحة في صورة المعاينة
                debug_montage[i*100:(i+1)*100, j*100:(j+1)*100] = canvas
                
                try:
                    # قراءة الرقم عبر Tesseract
                    text = pytesseract.image_to_string(canvas, config=custom_config)
                    for ch in text.strip():
                        if ch.isdigit() and ch != '0':
                            board[i][j] = int(ch)
                            break
                except Exception:
                    pass
            
        progress.progress((idx + 1) / total)
        time.sleep(0.01) # تحديث شريط التقدم
        
    progress.empty()
    st.session_state.debug_clean = debug_montage
    return board

# ==========================================
# 4. خوارزمية الحل والتحقق
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
                return False, f"قيمة خارج النطاق ({v}) في ({i + 1},{j + 1})"
            if v != 0:
                temp[i][j] = 0
                if not is_valid(temp, i, j, v):
                    temp[i][j] = v
                    return False, f"تعارض في قراءة الـ OCR (الرقم {v} مكرر بشكل خاطئ في الشبكة)"
                temp[i][j] = v
    return True, ""

# ==========================================
# 5. الواقع المعزز 
# ==========================================
def draw_solution(img, solved, original):
    ch = img.shape[0] // 9
    cw = img.shape[1] // 9
    for i in range(9):
        for j in range(9):
            if original[i][j] == 0 and solved[i][j] != 0:
                text = str(solved[i][j])
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                tx = j * cw + (cw - ts[0]) // 2
                ty = i * ch + (ch + ts[1]) // 2
                # رسم الأرقام باللون الأخضر
                cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

# ==========================================
# واجهة المستخدم
# ==========================================
st.title("🧩 حلّال السودوكو الذكي (محرك Tesseract)")
st.write("بدون إنترنت أو APIs خارجية! التقط صورة أو ارفع ملفاً للحل الفوري.")

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
            try:
                doc = fitz.open(stream=file.read(), filetype="pdf")
                if doc.page_count > 0:
                    pix = doc[0].get_pixmap(dpi=200)
                    cv2_img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
                doc.close()
            except Exception as e:
                st.error(f"خطأ في PDF: {e}")
        else:
            cv2_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

if cv2_img is not None:
    img_hash = hash(cv2_img.tobytes()[:8000])
    if st.session_state.img_hash != img_hash:
        st.session_state.img_hash = img_hash
        st.session_state.board_extracted = False
        st.session_state.original_board = None
        st.session_state.solved_img = None
        st.session_state.solved_board = None

    thresh = preprocess_image(cv2_img)
    contour = find_board(thresh)

    if contour is not None:
        SIZE = 450
        warped, warp_matrix = warp_image(cv2_img, contour, SIZE)

        # سير العمل التلقائي
        if not st.session_state.board_extracted:
            with st.spinner("⏳ جاري قراءة الأرقام محلياً وبناء الحل..."):
                board = smart_extract_digits(warped)
                
                if board is not None and np.count_nonzero(board) > 0:
                    st.session_state.original_board = board.copy()
                    ok, msg = validate_board(board.copy())
                    
                    if not ok:
                        st.error(f"❌ تم استخراج الأرقام لكن يوجد تعارض: {msg}")
                    else:
                        solved = board.copy()
                        if solve_sudoku(solved):
                            st.success("✅ تم التعرف على الأرقام وإيجاد الحل بنجاح!")
                            
                            ar = np.zeros((SIZE, SIZE, 3), np.uint8)
                            ar = draw_solution(ar, solved, board)
                            inv_M = cv2.getPerspectiveTransform(
                                np.float32([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]]),
                                order_points(contour)
                            )
                            inv_warp = cv2.warpPerspective(ar, inv_M, (cv2_img.shape[1], cv2_img.shape[0]))
                            result = cv2.addWeighted(cv2_img, 1, inv_warp, 1, 0)
                            
                            st.session_state.solved_img = result
                            st.session_state.solved_board = solved
                            st.balloons()
                        else:
                            st.error("⚠️ لم نتمكن من إيجاد حل رياضي صحيح لهذه الشبكة.")
                else:
                    st.error("❌ لم نتمكن من العثور على أرقام واضحة.")
                    
            st.session_state.board_extracted = True

        # عرض النتيجة النهائية
        if st.session_state.solved_img is not None:
            st.image(st.session_state.solved_img, channels="BGR", caption="✨ الحل التلقائي", use_container_width=True)
            with st.expander("📊 عرض الحل كجدول"):
                st.dataframe(pd.DataFrame(st.session_state.solved_board, columns=[f"C{i}" for i in range(1, 10)], index=[f"R{i}" for i in range(1, 10)]), use_container_width=True)
                
        if st.session_state.debug_clean is not None:
            with st.expander("🛠️ كيف رأى Tesseract الأرقام؟ (أرقام سوداء معزولة)"):
                st.image(st.session_state.debug_clean, caption="اللوحة المجمعة التي تمت قراءتها", use_container_width=True)

    else:
        st.error("❌ لم يتم العثور على شبكة سودوكو واضحة في الصورة.")
