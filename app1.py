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
    'warped_img': None,
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
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل النموذج: {e}")
        return None

# ==========================================
# 1. دوال معالجة الصور الأساسية
# ==========================================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1)
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

def find_board(thresh_img):
    contours, _ = cv2.findContours(
        thresh_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
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
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def warp_image(img, pts, size=450):
    src = order_points(pts)
    dst = np.float32([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size)), M

# ==========================================
# 2. استخراج الأرقام بالذكاء الاصطناعي
# ==========================================
def smart_extract_digits_cnn(warped_img, model):
    board = np.zeros((9, 9), dtype=int)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    debug_montage = np.zeros((9 * 28, 9 * 28), dtype=np.uint8)
    progress = st.progress(0, text="🤖 الذكاء الاصطناعي يحلل الأرقام...")
    cell_size = 450 // 9
    detected_count = 0
    for i in range(9):
        for j in range(9):
            m = int(cell_size * 0.12)
            y1, y2 = i * cell_size + m, (i + 1) * cell_size - m
            x1, x2 = j * cell_size + m, (j + 1) * cell_size - m
            cell = thresh[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            contours, _ = cv2.findContours(
                cell,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 30:
                    x, y, w, h = cv2.boundingRect(c)
                    digit = cell[y:y + h, x:x + w]
                    if digit.size == 0 or w < 3 or h < 3:
                        continue
                    # تجهيز 28x28 (MNIST Style)
                    canvas = np.zeros((28, 28), dtype=np.uint8)
                    scale = 20.0 / max(w, h)
                    nw = max(int(w * scale), 1)
                    nh = max(int(h * scale), 1)
                    res = cv2.resize(
                        digit,
                        (nw, nh),
                        interpolation=cv2.INTER_AREA
                    )
                    oy = (28 - nh) // 2
                    ox = (28 - nw) // 2
                    canvas[oy:oy + nh, ox:ox + nw] = res
                    debug_montage[
                        i * 28:(i + 1) * 28,
                        j * 28:(j + 1) * 28
                    ] = canvas
                    # التنبؤ
                    inp = canvas.reshape(1, 28, 28, 1).astype('float32') / 255.0
                    try:
                        pred = model.predict(inp, verbose=0)
                        confidence = np.max(pred)
                        digit_val = np.argmax(pred)
                        if confidence > 0.7 and digit_val != 0:
                            board[i][j] = digit_val
                            detected_count += 1
                    except Exception as e:
                        st.warning(f"خطأ في التنبؤ [{i},{j}]: {e}")
            progress.progress(((i * 9 + j) + 1) / 81)
    progress.empty()
    st.session_state.debug_clean = debug_montage
    # ✅ إصلاح: إظهار عدد الأرقام المكتشفة
    st.info(f"🔢 تم اكتشاف {detected_count} رقم من 81 خلية")
    return board

# ==========================================
# 3. محرك حل السودوكو
# ==========================================
def is_valid(b, r, c, n):
    if n in b[r, :] or n in b[:, c]:
        return False
    sr, sc = (r // 3) * 3, (c // 3) * 3
    if n in b[sr:sr + 3, sc:sc + 3]:
        return False
    return True

def solve(b):
    for r in range(9):
        for c in range(9):
            if b[r, c] == 0:
                for n in range(1, 10):
                    if is_valid(b, r, c, n):
                        b[r, c] = n
                        if solve(b):
                            return True
                        b[r, c] = 0
                return False
    return True

def validate_board(board):
    for i in range(9):
        for j in range(9):
            v = board[i, j]
            if v != 0:
                temp = board.copy()
                temp[i, j] = 0
                if not is_valid(temp, i, j, v):
                    return False, f"تكرار الرقم {v} في [{i+1},{j+1}]"
    return True, ""

# ==========================================
# 4. رسم الحل على الصورة الأصلية
# ==========================================
def draw_solution_on_warped(warped_img, solved, original):
    """رسم الأرقام المحلولة على الصورة المقصوصة مباشرة"""
    result = warped_img.copy()
    h, w = result.shape[:2]
    ch, cw = h // 9, w // 9
    for i in range(9):
        for j in range(9):
            if original[i, j] == 0 and solved[i, j] != 0:
                txt = str(solved[i, j])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = 1.2
                thick = 2
                size = cv2.getTextSize(txt, font, fs, thick)[0]
                tx = j * cw + (cw - size[0]) // 2
                ty = i * ch + (ch + size[1]) // 2
                # ✅ خلفية بيضاء + رقم أحمر لوضوح أفضل
                pad = 4
                cv2.rectangle(
                    result,
                    (tx - pad, ty - size[1] - pad),
                    (tx + size[0] + pad, ty + pad),
                    (255, 255, 255),
                    -1
                )
                cv2.putText(
                    result,
                    txt,
                    (tx, ty),
                    font,
                    fs,
                    (0, 0, 255),
                    thick
                )
    return result

def overlay_solution_on_original(original_img, solved_warped, pts, size=450):
    """عكس الصورة المحلولة على الصورة الأصلية"""
    src = np.float32([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ])
    dst = order_points(pts)
    M_inv = cv2.getPerspectiveTransform(src, dst)
    h, w = original_img.shape[:2]
    warped_back = cv2.warpPerspective(solved_warped, M_inv, (w, h))
    # إنشاء قناع للمنطقة المحلولة
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:] = 255
    mask_warped = cv2.warpPerspective(mask, M_inv, (w, h))
    # دمج الصورة الأصلية مع الحل
    result = original_img.copy()
    mask_3ch = cv2.cvtColor(mask_warped, cv2.COLOR_GRAY2BGR)
    # حيث يوجد القناع، استخدم الصورة المحلولة
    result = np.where(mask_3ch > 0, warped_back, result)
    return result

# ==========================================
# واجهة التطبيق
# ==========================================
st.title("🤖 حلّال السودوكو الذكي النهائي")
st.caption("يستخدم نموذج CNN مخصص لرؤية الأرقام وحلها آلياً")

model = load_digit_model()
if model is None:
    st.error("⚠️ ملف model.h5 مفقود أو تالف!")
    st.stop()
st.success("✅ تم تحميل النموذج بنجاح")

choice = st.radio("المصدر:", ("📸 كاميرا", "📁 ملف"), horizontal=True)

if choice == "📸 كاميرا":
    upload = st.camera_input("صوّر اللغز")
else:
    upload = st.file_uploader(
        "ارفع صورة",
        type=['png', 'jpg', 'jpeg', 'pdf']
    )

if upload:
    data = upload.getvalue()

    # معالجة PDF
    if upload.name.lower().endswith('.pdf'):
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=200)
        img = cv2.imdecode(
            np.frombuffer(pix.tobytes("png"), np.uint8),
            1
        )
        doc.close()
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

    if img is None:
        st.error("❌ لم يتم قراءة الصورة!")
        st.stop()

    # ✅ إصلاح: عرض الصورة الأصلية
    st.subheader("📷 الصورة المرفوعة")
    st.image(img, channels="BGR", use_container_width=True)

    # التحقق من تغيير الصورة
    h_val = hashlib.md5(img.tobytes()[:5000]).hexdigest()
    if st.session_state.img_hash != h_val:
        st.session_state.update({
            'img_hash': h_val,
            'board_extracted': False,
            'extracted_board': None,
            'solved_img': None,
            'solved_board': None,
            'debug_clean': None,
            'warped_img': None,
        })

    thresh = preprocess_image(img)
    pts = find_board(thresh)
    if pts is None:
        st.error("❌ لم يتم العثور على شبكة سودوكو في الصورة!")
        st.info("💡 تأكد أن الصورة واضحة وتحتوي على شبكة كاملة")
        st.stop()

    warped, M = warp_image(img, pts)
    st.session_state.warped_img = warped.copy()

    # ✅ إصلاح: عرض الصورة المقتطعة
    st.subheader("🔲 الشبكة المكتشفة")
    st.image(warped, channels="BGR", use_container_width=True)

    # === استخراج الأرقام ===
    if not st.session_state.board_extracted:
        with st.spinner("🤖 جاري تحليل الأرقام..."):
            board = smart_extract_digits_cnn(warped, model)
            st.session_state.extracted_board = board.copy()
            st.session_state.board_extracted = True

    # === عرض اللوحة المكتشفة ===
    if st.session_state.board_extracted:
        st.subheader("🔢 اللوحة المكتشفة")
        b = st.session_state.extracted_board.copy()
        ok, msg = validate_board(b)
        if not ok:
            st.warning(f"⚠️ خطأ: {msg}")
        st.info("👇 صحح الأرقام في الجدول ثم اضغط 'حل الآن'")

        # ✅ إصلاح: عرض الجدول دائماً للتعديل
        df = pd.DataFrame(
            b,
            columns=[f"C{i+1}" for i in range(9)],
            index=[f"R{i+1}" for i in range(9)]
        )
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            key="sudoku_editor"
        )

        # === زر الحل ===
        if st.button("🚀 حل السودوكو", type="primary", use_container_width=True):
            final_board = edited_df.to_numpy().astype(int)
            # التحقق من الصحة
            ok2, msg2 = validate_board(final_board)
            if not ok2:
                st.error(f"❌ اللوحة غير صحيحة: {msg2}")
            else:
                # عدد الأرقام
                clue_count = np.count_nonzero(final_board)
                if clue_count < 17:
                    st.warning(
                        f"⚠️ عدد الأرقام ({clue_count}) قليل جداً. "
                        f"الحد الأدنى 17 رقم."
                    )
                s_board = final_board.copy()
                original_board = final_board.copy()
                with st.spinner("⏳ جاري حل اللغز..."):
                    if solve(s_board):
                        st.success("✅ تم حل السودوكو بنجاح!")
                        st.balloons()
                        st.session_state.solved_board = s_board.copy()

                        # رسم الحل على الصورة المقتطعة
                        warped_copy = st.session_state.warped_img.copy()
                        solved_warped = draw_solution_on_warped(
                            warped_copy,
                            s_board,
                            original_board
                        )

                        # عكس على الصورة الأصلية
                        result = overlay_solution_on_original(
                            img,
                            solved_warped,
                            pts
                        )
                        st.session_state.solved_img = result
                    else:
                        # ✅ إصلاح: رسالة خطأ عند فشل الحل
                        st.error(
                            "❌ لا يمكن حل هذا اللغز! "
                            "تأكد من صحة الأرقام المدخلة."
                        )

        # === عرض النتيجة ===
        if st.session_state.solved_img is not None:
            st.subheader("🎯 النتيجة النهائية")
            st.image(
                st.session_state.solved_img,
                channels="BGR",
                use_container_width=True
            )

            # عرض الحل كجدول
            if st.session_state.solved_board is not None:
                with st.expander("📊 الحل كجدول"):
                    sol_df = pd.DataFrame(
                        st.session_state.solved_board,
                        columns=[f"C{i+1}" for i in range(9)],
                        index=[f"R{i+1}" for i in range(9)]
                    )
                    st.dataframe(sol_df, use_container_width=True)

        # === المعاينة التقنية ===
        if st.session_state.debug_clean is not None:
            with st.expander("🛠️ المعاينة التقنية"):
                st.image(
                    st.session_state.debug_clean,
                    caption="الأرقام كما رآها الذكاء الاصطناعي",
                    use_container_width=True
    )
