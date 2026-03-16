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
    """تحويل الصورة إلى أبيض وأسود مع تحسين التباين"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    return thresh

def find_board(thresh_img):
    """البحث عن أكبر مربع في الصورة (شبكة السودوكو)"""
    contours, _ = cv2.findContours(
        thresh_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
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
    """ترتيب نقاط المربع: أعلى-يسار، أعلى-يمين، أسفل-يمين، أسفل-يسار"""
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
    """تصحيح المنظور لاستخراج الشبكة بشكل مستقيم"""
    src = order_points(pts)
    dst = np.float32([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, M

# ==========================================
# 2. كشف الخلايا الممتلئة (بدون API)
# ==========================================
def get_cell_inner(gray, row, col, grid_size=450, margin=0.20):
    """استخراج المنطقة الداخلية لخلية محددة (بدون خطوط الشبكة)"""
    cell_size = grid_size // 9
    y1 = row * cell_size
    x1 = col * cell_size
    cell = gray[y1:y1 + cell_size, x1:x1 + cell_size]
    # قص الحواف لإزالة خطوط الشبكة
    m = int(cell_size * margin)
    inner = cell[m:cell_size - m, m:cell_size - m]
    return inner

def is_cell_filled(cell_gray):
    """ تحديد ما إذا كانت الخلية تحتوي على رقم باستخدام تحليل البكسلات + الكونتورات """
    _, thresh = cv2.threshold(
        cell_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    total = thresh.shape[0] * thresh.shape[1]
    if total == 0:
        return False
    white = cv2.countNonZero(thresh)
    ratio = white / total
    # نسبة قليلة جداً = خلية فارغة
    if ratio < 0.03:
        return False
    # نسبة عالية جداً = ربما ضوضاء أو خطوط
    if ratio > 0.80:
        return False
    # فحص إضافي: هل يوجد كونتور يشبه الرقم؟
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        area = cv2.contourArea(c)
        if area > total * 0.03:
            _, _, cw, ch = cv2.boundingRect(c)
            if cw > 0 and 0.3 < (ch / cw) < 5.0:
                return True
    return False

def detect_filled_cells(warped_img):
    """إنشاء مصفوفة 9×9 توضح أي الخلايا ممتلئة"""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((9, 9), dtype=bool)
    for i in range(9):
        for j in range(9):
            inner = get_cell_inner(gray, i, j)
            mask[i][j] = is_cell_filled(inner)
    return mask

# ==========================================
# 3. إزالة خطوط الشبكة (تحسين قبل OCR)
# ==========================================
def remove_grid_lines(warped_img):
    """ إزالة الخطوط الأفقية والعمودية من الصورة لتحسين دقة OCR """
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    # كشف الخطوط الأفقية
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    # كشف الخطوط العمودية
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    # إزالة الخطوط
    all_lines = cv2.add(h_lines, v_lines)
    clean = cv2.subtract(thresh, all_lines)
    # ترميم الأرقام المتضررة
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, repair_kernel)
    # قلب الألوان: أرقام سوداء على خلفية بيضاء
    result = cv2.bitwise_not(clean)
    return result

# ==========================================
# 4. استخراج الأرقام عبر OCR
# ==========================================
def ocr_overlay_method(warped_img):
    """ ★ الطريقة الأساسية ★ استخدام OCR مع Overlay للحصول على إحداثيات كل رقم ثم ربطه بالخلية الصحيحة بناءً على موقعه """
    board = np.zeros((9, 9), dtype=int)
    h, w = warped_img.shape[:2]
    cell_h = h / 9.0
    cell_w = w / 9.0
    # تنظيف الصورة من خطوط الشبكة
    clean = remove_grid_lines(warped_img)
    # تحويل إلى base64
    _, buf = cv2.imencode('.png', clean)
    b64 = base64.b64encode(buf).decode('utf-8')
    payload = {
        'apikey': API_KEY,
        'base64Image': f'data:image/png;base64,{b64}',
        'OCREngine': '2',
        'isOverlayRequired': 'true',  # ★ المفتاح: طلب الإحداثيات
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
                    # استخراج كل رقم مع حساب موقعه
                    for ci, ch in enumerate(text):
                        if ch.isdigit() and ch != '0':
                            val = int(ch)
                            if 1 <= val <= 9:
                                # حساب مركز الحرف
                                if len(text) == 1:
                                    cx = left + ww / 2
                                else:
                                    cx = left + (ci + 0.5) * ww / len(text)
                                cy = top + wh / 2
                                # تحديد الصف والعمود
                                col = min(8, max(0, int(cx / cell_w)))
                                row = min(8, max(0, int(cy / cell_h)))
                                board[row][col] = val
            st.session_state.debug_clean = clean
            return board, None
        # إذا لم يتوفر Overlay، محاولة Engine 1
        return try_engine1_overlay(warped_img)
    except requests.exceptions.Timeout:
        return None, "انتهت مهلة الاتصال"
    except Exception as e:
        return None, str(e)

def try_engine1_overlay(warped_img):
    """محاولة باستخدام Engine 1 الذي يدعم Overlay بشكل أفضل"""
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
        'OCREngine': '1',  # Engine 1
        'isOverlayRequired': 'true',
        'scale': 'true',
    }
    try:
        resp = requests.post(OCR_URL, data=payload, timeout=30)
        data = resp.json()
        parsed = data.get('ParsedResults', [])
        if not parsed:
            return None, "لا نتائج من Engine 1"
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
            return board, None
        return None, "لا يوجد Overlay"
    except Exception as e:
        return None, str(e)

def ocr_cell_by_cell(warped_img, filled_mask):
    """ ★ الطريقة الاحتياطية ★ إرسال كل خلية ممتلئة بشكل منفرد إلى OCR أبطأ لكن أكثر دقة """
    board = np.zeros((9, 9), dtype=int)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    # جمع الخلايا الممتلئة
    cells_to_process = []
    for i in range(9):
        for j in range(9):
            if filled_mask[i][j]:
                inner = get_cell_inner(gray, i, j)
                cells_to_process.append((i, j, inner))
    if not cells_to_process:
        return board
    total = len(cells_to_process)
    progress = st.progress(0, text=f"التعرف على {total} خلية...")
    for idx, (i, j, cell_img) in enumerate(cells_to_process):
        # تحضير صورة الخلية
        resized = cv2.resize(cell_img, (100, 100), interpolation=cv2.INTER_CUBIC)
        # إضافة حواف بيضاء
        padded = cv2.copyMakeBorder(
            resized,
            25, 25, 25, 25,
            cv2.BORDER_CONSTANT,
            value=255
        )
        # تحسين التباين
        _, binary = cv2.threshold(
            padded,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, buf = cv2.imencode('.png', binary)
        b64 = base64.b64encode(buf).decode('utf-8')
        payload = {
            'apikey': API_KEY,
            'base64Image': f'data:image/png;base64,{b64}',
            'OCREngine': '2',
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
        progress.progress(
            (idx + 1) / total,
            text=f"خلية ({i + 1},{j + 1}) — {idx + 1}/{total}"
        )
        time.sleep(0.15)  # تجنب تجاوز حد الطلبات
    progress.empty()
    return board

def smart_extract_digits(warped_img):
    """ ★ الدالة الرئيسية ★ تجرب Overlay أولاً، ثم تلجأ لخلية-بخلية إذا فشلت """
    # الخطوة 1: كشف الخلايا الممتلئة (بدون API)
    filled_mask = detect_filled_cells(warped_img)
    expected_count = int(np.sum(filled_mask))
    st.session_state.filled_mask = filled_mask
    st.caption(f"🔍 تم اكتشاف **{expected_count}** خلية تحتوي على أرقام")
    if expected_count == 0:
        st.error("لم يتم العثور على أي أرقام في الشبكة!")
        return None
    # الخطوة 2: محاولة Overlay (استدعاء API واحد)
    board, error = ocr_overlay_method(warped_img)
    if board is not None:
        detected = int(np.count_nonzero(board))
        # تحقق: هل اكتشف OCR عدداً معقولاً؟
        if detected >= expected_count * 0.5:
            st.caption(f"✅ Overlay: تم التعرف على **{detected}** رقم")
            # مطابقة مع كشف الخلايا: إذا اكتشف OCR رقماً
            # في خلية فارغة حسب تحليل البكسلات، نتجاهله
            for i in range(9):
                for j in range(9):
                    if board[i][j] != 0 and not filled_mask[i][j]:
                        board[i][j] = 0  # رقم وهمي
            return board
        else:
            st.warning(
                f"⚠️ Overlay ضعيف ({detected}/{expected_count}). "
                f"جاري التجربة خلية بخلية..."
            )
    elif error:
        st.warning(f"⚠️ Overlay فشل: {error}. جاري التجربة خلية بخلية...")
    # الخطوة 3: احتياطي — خلية بخلية
    board = ocr_cell_by_cell(warped_img, filled_mask)
    detected = int(np.count_nonzero(board))
    st.caption(f"🔄 Cell-by-cell: تم التعرف على **{detected}** رقم")
    return board

# ==========================================
# 5. خوارزمية الحل (Backtracking)
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
    """التحقق من عدم وجود تكرارات في الشبكة"""
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
                    return False, f"تكرار الرقم {v} في ({i + 1},{j + 1})"
                temp[i][j] = v
    return True, ""

# ==========================================
# 6. الواقع المعزز (AR Overlay)
# ==========================================
def draw_solution(img, solved, original):
    """رسم الأرقام المحلولة فقط (الخلايا التي كانت فارغة)"""
    ch = img.shape[0] // 9
    cw = img.shape[1] // 9
    for i in range(9):
        for j in range(9):
            if original[i][j] == 0 and solved[i][j] != 0:
                text = str(solved[i][j])
                ts = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    2
                )[0]
                tx = j * cw + (cw - ts[0]) // 2
                ty = i * ch + (ch + ts[1]) // 2
                cv2.putText(
                    img,
                    text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
    return img

# ==========================================
# واجهة المستخدم
# ==========================================
st.title("🧩 حلّال السودوكو الذكي بالواقع المعزز")
st.write("التقط صورة أو ارفع ملفاً وسنحل لك اللغز!")

# اختيار طريقة الإدخال
method = st.radio(
    "طريقة الإدخال:",
    ("📸 كاميرا", "📁 ملف (صورة/PDF)", "✏️ إدخال يدوي"),
    key="input_method",
    horizontal=True
)

cv2_img = None

# === الكاميرا ===
if method == "📸 كاميرا":
    cam = st.camera_input("التقط صورة للسودوكو", key="cam")
    if cam:
        cv2_img = cv2.imdecode(
            np.frombuffer(cam.getvalue(), np.uint8),
            cv2.IMREAD_COLOR
        )

# === رفع ملف ===
elif method == "📁 ملف (صورة/PDF)":
    file = st.file_uploader(
        "ارفع صورة أو PDF",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        key="uploader"
    )
    if file:
        if file.name.lower().endswith('.pdf'):
            try:
                doc = fitz.open(stream=file.read(), filetype="pdf")
                if doc.page_count > 0:
                    pix = doc[0].get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    cv2_img = cv2.imdecode(
                        np.frombuffer(img_bytes, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                else:
                    st.error("ملف PDF فارغ!")
                doc.close()
            except Exception as e:
                st.error(f"خطأ في PDF: {e}")
        else:
            raw = file.read()
            cv2_img = cv2.imdecode(
                np.frombuffer(raw, np.uint8),
                cv2.IMREAD_COLOR
            )

# === إدخال يدوي ===
elif method == "✏️ إدخال يدوي":
    st.info("أدخل الأرقام (0 = خلية فارغة)")
    empty_grid = np.zeros((9, 9), dtype=int)
    df = pd.DataFrame(
        empty_grid,
        columns=[f"C{i}" for i in range(1, 10)]
    )
    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        key="manual_grid"
    )
    if st.button("🚀 حل", type="primary", use_container_width=True, key="manual_solve"):
        try:
            board = edited.to_numpy().astype(int)
        except (ValueError, TypeError):
            st.error("أدخل أرقاماً صحيحة فقط (0-9)")
            st.stop()
        ok, msg = validate_board(board.copy())
        if not ok:
            st.error(f"❌ {msg}")
            st.stop()
        solved = board.copy()
        if solve_sudoku(solved):
            st.success("✅ تم الحل!")
            st.dataframe(
                pd.DataFrame(solved, columns=[f"C{i}" for i in range(1, 10)]),
                use_container_width=True
            )
            st.balloons()
        else:
            st.error("⚠️ لا يوجد حل لهذه الشبكة")

# ==========================================
# معالجة الصورة (كاميرا أو ملف)
# ==========================================
if cv2_img is not None:
    # فحص تغيّر الصورة
    img_hash = hash(cv2_img.tobytes()[:8000])
    if st.session_state.img_hash != img_hash:
        st.session_state.img_hash = img_hash
        st.session_state.board_extracted = False
        st.session_state.original_board = None
        st.session_state.solved_img = None
        st.session_state.solved_board = None
        st.session_state.filled_mask = None

    # معالجة الصورة
    thresh = preprocess_image(cv2_img)
    contour = find_board(thresh)

    if contour is not None:
        SIZE = 450
        warped, warp_matrix = warp_image(cv2_img, contour, SIZE)

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                warped,
                channels="BGR",
                caption="الشبكة المكتشفة",
                use_container_width=True
            )

        # استخراج الأرقام مرة واحدة
        if not st.session_state.board_extracted:
            with st.spinner("⏳ جاري تحليل الشبكة والتعرف على الأرقام..."):
                board = smart_extract_digits(warped)
                if board is not None:
                    st.session_state.original_board = board.copy()
                    st.session_state.board_extracted = True

        with col2:
            if (st.session_state.board_extracted and st.session_state.original_board is not None):
                st.markdown("### 📝 مراجعة وتعديل")
                st.info("0 = فارغة. عدّل أي خلية قرأها OCR بشكل خاطئ.")
                df = pd.DataFrame(
                    st.session_state.original_board,
                    columns=[f"C{i}" for i in range(1, 10)]
                )
                edited_df = st.data_editor(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    key="grid_editor"
                )

                # عرض خريطة الخلايا المكتشفة
                if st.session_state.filled_mask is not None:
                    with st.expander("🔍 خريطة الخلايا المكتشفة"):
                        mask_display = st.session_state.filled_mask.astype(int)
                        st.dataframe(
                            pd.DataFrame(
                                mask_display,
                                columns=[f"C{i}" for i in range(1, 10)]
                            ),
                            use_container_width=True
                        )
                        st.caption("1 = خلية ممتلئة، 0 = خلية فارغة")

                if st.button("🚀 تأكيد وحل اللغز", type="primary", use_container_width=True, key="solve_btn"):
                    try:
                        corrected = edited_df.to_numpy().astype(int)
                    except (ValueError, TypeError):
                        st.error("أدخل أرقاماً صحيحة فقط (0-9)")
                        st.stop()
                    ok, msg = validate_board(corrected.copy())
                    if not ok:
                        st.error(f"❌ {msg}")
                        st.stop()
                    solved = corrected.copy()
                    with st.spinner("🧠 جاري الحل..."):
                        if solve_sudoku(solved):
                            st.success("✅ تم إيجاد الحل!")
                            # بناء الواقع المعزز
                            ar = np.zeros((SIZE, SIZE, 3), np.uint8)
                            ar = draw_solution(ar, solved, corrected)
                            inv_M = cv2.getPerspectiveTransform(
                                np.float32([
                                    [0, 0],
                                    [SIZE - 1, 0],
                                    [SIZE - 1, SIZE - 1],
                                    [0, SIZE - 1]
                                ]),
                                order_points(contour)
                            )
                            inv_warp = cv2.warpPerspective(
                                ar,
                                inv_M,
                                (cv2_img.shape[1], cv2_img.shape[0])
                            )
                            result = cv2.addWeighted(
                                cv2_img,
                                1,
                                inv_warp,
                                1,
                                0
                            )
                            st.session_state.solved_img = result
                            st.session_state.solved_board = solved
                            st.balloons()
                        else:
                            st.error(
                                "⚠️ لا يوجد حل. تأكد من صحة الأرقام."
                            )

        # عرض النتيجة النهائية
        if st.session_state.solved_img is not None:
            st.markdown("---")
            st.markdown("### 🎯 النتيجة النهائية")
            st.image(
                st.session_state.solved_img,
                channels="BGR",
                caption="الحل بالواقع المعزز",
                use_container_width=True
            )
            if st.session_state.solved_board is not None:
                st.markdown("### 📊 الحل كجدول")
                st.dataframe(
                    pd.DataFrame(
                        st.session_state.solved_board,
                        columns=[f"C{i}" for i in range(1, 10)],
                        index=[f"R{i}" for i in range(1, 10)]
                    ),
                    use_container_width=True
                )

        # عرض الصورة المنظفة (تصحيح أخطاء)
        if st.session_state.debug_clean is not None:
            with st.expander("🛠️ الصورة بعد إزالة خطوط الشبكة"):
                st.image(
                    st.session_state.debug_clean,
                    caption="الصورة المنظفة المرسلة إلى OCR",
                    use_container_width=True
                )
    else:
        st.error("❌ لم يتم العثور على شبكة سودوكو واضحة")
        st.info(
            "💡 نصائح:\n"
            "- تأكد أن الشبكة كاملة وواضحة في الصورة\n"
            "- حاول التصوير من زاوية مستقيمة\n"
            "- تأكد من الإضاءة الجيدة\n"
            "- أو استخدم **الإدخال اليدوي** كبديل"
        )
