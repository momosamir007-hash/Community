import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
import fitz
import time

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Sudoku AR Solver",
    page_icon="🧩",
    layout="centered"
)

# ==========================================
# إعدادات API
# ==========================================
try:
    API_KEY = st.secrets["OCR_API_KEY"]
except KeyError:
    st.error("⚠️ لم يتم العثور على مفتاح OCR_API_KEY في secrets")
    st.stop()

OCR_URL = 'https://api.ocr.space/parse/image'

# ==========================================
# تهيئة session_state
# ==========================================
for key, default in {
    'img_hash': None,
    'board_extracted': False,
    'original_board': None,
    'solved_img': None,
    'solved_board': None,
    'filled_mask': None,
    'debug_clean': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 1. دوال معالجة الصور
# ==========================================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
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
# 2. كشف الخلايا الممتلئة 
# ==========================================
def get_cell_inner(gray, row, col, grid_size=450, margin=0.15):
    cell_size = grid_size // 9
    y1 = row * cell_size
    x1 = col * cell_size
    cell = gray[y1:y1 + cell_size, x1:x1 + cell_size]
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
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > total * 0.03:
            _, _, cw, ch = cv2.boundingRect(c)
            if cw > 0 and 0.2 < (ch / cw) < 6.0:
                return True
    return False

def detect_filled_cells(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((9, 9), dtype=bool)
    for i in range(9):
        for j in range(9):
            inner = get_cell_inner(gray, i, j)
            mask[i][j] = is_cell_filled(inner)
    return mask

# ==========================================
# 3. إزالة خطوط الشبكة 
# ==========================================
def remove_grid_lines(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    all_lines = cv2.add(h_lines, v_lines)
    clean = cv2.subtract(thresh, all_lines)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, repair_kernel)
    result = cv2.bitwise_not(clean)
    return result

# ==========================================
# 4. استخراج الأرقام عبر OCR
# ==========================================
def ocr_overlay_method(warped_img):
    board = np.zeros((9, 9), dtype=int)
    h, w = warped_img.shape[:2]
    cell_h = h / 9.0
    cell_w = w / 9.0
    clean = remove_grid_lines(warped_img)
    _, buf = cv2.imencode('.png', clean)
    b64 = base64.b64encode(buf).decode('utf-8')
    payload = {
        'apikey': API_KEY,
        'base64Image': f'data:image/png;base64,{b64}',
        'OCREngine': '1', # تم التغيير إلى المحرك 1 ليكون أدق
        'isOverlayRequired': 'true',
        'scale': 'true',
    }
    try:
        resp = requests.post(OCR_URL, data=payload, timeout=30)
        data = resp.json()
        if data.get('IsErroredOnProcessing'):
            return None, f"خطأ OCR: {data.get('ErrorMessage')}"
        parsed = data.get('ParsedResults', [])
        if not parsed:
            return None, "لا توجد نتائج"
        overlay = parsed[0].get('TextOverlay')
        if overlay and overlay.get('Lines'):
            for line in overlay['Lines']:
                for word in line.get('Words', []):
                    text = word.get('WordText', '')
                    left = word.get('Left', 0)
                    top = word.get('Top', 0)
                    ww = max(word.get('Width', 1), 1)
                    wh = max(word.get('Height', 1), 1)
                    for ci, ch in enumerate(text):
                        if ch.isdigit() and ch != '0':
                            val = int(ch)
                            if 1 <= val <= 9:
                                if len(text) == 1:
                                    cx = left + ww / 2
                                else:
                                    cx = left + (ci + 0.5) * ww / len(text)
                                cy = top + wh / 2
                                col = min(8, max(0, int(cx / cell_w)))
                                row = min(8, max(0, int(cy / cell_h)))
                                board[row][col] = val
            st.session_state.debug_clean = clean
            return board, None
        return None, "لا يوجد Overlay"
    except Exception as e:
        return None, str(e)

def ocr_cell_by_cell(warped_img, filled_mask):
    board = np.zeros((9, 9), dtype=int)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    cells_to_process = []
    for i in range(9):
        for j in range(9):
            if filled_mask[i][j]:
                inner = get_cell_inner(gray, i, j)
                cells_to_process.append((i, j, inner))
    if not cells_to_process:
        return board
    total = len(cells_to_process)
    progress = st.progress(0, text=f"استخراج دقيق ({total} خلية)...")
    for idx, (i, j, cell_img) in enumerate(cells_to_process):
        resized = cv2.resize(cell_img, (100, 100), interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(resized, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
        _, binary = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, buf = cv2.imencode('.png', binary)
        b64 = base64.b64encode(buf).decode('utf-8')
        payload = {
            'apikey': API_KEY,
            'base64Image': f'data:image/png;base64,{b64}',
            'OCREngine': '1', # Engine 1 fallback
            'scale': 'true',
        }
        try:
            resp = requests.post(OCR_URL, data=payload, timeout=10)
            result = resp.json()
            if result.get('ParsedResults'):
                txt = result['ParsedResults'][0].get('ParsedText', '')
                for ch in txt.strip():
                    if ch.isdigit() and ch != '0':
                        board[i][j] = int(ch)
                        break
        except Exception:
            pass
        progress.progress((idx + 1) / total)
        time.sleep(0.15)
    progress.empty()
    return board

def smart_extract_digits(warped_img):
    filled_mask = detect_filled_cells(warped_img)
    expected_count = int(np.sum(filled_mask))
    st.session_state.filled_mask = filled_mask
    if expected_count == 0:
        return None
    
    board, error = ocr_overlay_method(warped_img)
    if board is not None:
        detected = int(np.count_nonzero(board))
        if detected >= expected_count * 0.5:
            for i in range(9):
                for j in range(9):
                    if board[i][j] != 0 and not filled_mask[i][j]:
                        board[i][j] = 0
            return board
    board = ocr_cell_by_cell(warped_img, filled_mask)
    return board

# ==========================================
# 5. خوارزمية الحل 
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
                    return False, f"يوجد تكرار للرقم {v} في الصورة الأصلية"
                temp[i][j] = v
    return True, ""

# ==========================================
# 6. الواقع المعزز 
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
                cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

# ==========================================
# واجهة المستخدم
# ==========================================
st.title("🧩 حلّال السودوكو الذكي بالواقع المعزز (آلي بالكامل)")
st.write("التقط صورة أو ارفع ملفاً وسيقوم البرنامج بقراءتها وحلها فوراً!")

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

        # سير العمل التلقائي بالكامل
        if not st.session_state.board_extracted:
            with st.spinner("⏳ جاري تحليل الصورة واستخراج الأرقام لحلها..."):
                board = smart_extract_digits(warped)
                
                if board is not None:
                    st.session_state.original_board = board.copy()
                    ok, msg = validate_board(board.copy())
                    
                    if not ok:
                        st.error(f"❌ لم تنجح الأتمتة بالكامل بسبب جودة الصورة: {msg}")
                    else:
                        solved = board.copy()
                        if solve_sudoku(solved):
                            st.success("✅ تم التعرف على الأرقام وإيجاد الحل فوراً!")
                            
                            # رسم الواقع المعزز مباشرة
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
                            st.error("⚠️ لم نتمكن من إيجاد حل. ربما قرأ الـ OCR رقماً خاطئاً بسبب دقة الصورة.")
                else:
                    st.error("❌ لم نتمكن من استخراج أي أرقام من الشبكة.")
                    
            st.session_state.board_extracted = True

        # عرض النتيجة النهائية فقط
        if st.session_state.solved_img is not None:
            st.image(st.session_state.solved_img, channels="BGR", caption="✨ الحل التلقائي", use_container_width=True)
            with st.expander("📊 عرض الحل كجدول"):
                st.dataframe(pd.DataFrame(st.session_state.solved_board, columns=[f"C{i}" for i in range(1, 10)], index=[f"R{i}" for i in range(1, 10)]), use_container_width=True)

    else:
        st.error("❌ لم يتم العثور على شبكة سودوكو واضحة في الصورة.")
