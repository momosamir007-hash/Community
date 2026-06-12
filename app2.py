import streamlit as st
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import time
import tensorflow as tf
import hashlib
from io import BytesIO

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Sudoku AI Master Solver",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# تهيئة Session State
# ==========================================
defaults = {
    'img_hash': None,
    'board_extracted': False,
    'extracted_board': None,
    'confidences': None,
    'solved_img': None,
    'solved_board': None,
    'debug_clean': None,
    'warped_img': None,
    'original_img': None,
    'pts': None,
    'history': [],
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 0. تحميل النموذج
# ==========================================
@st.cache_resource
def load_digit_model():
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        return None

# ==========================================
# 1. دوال معالجة الصور
# ==========================================
def resize_if_needed(img, max_size=1500, min_size=300):
    """ضبط حجم الصورة تلقائياً"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    elif max(h, w) < min_size:
        scale = 600 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

def preprocess_image(img, clip_limit=2.0, block_size=11, c_val=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1)
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_val
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

def find_board_hough(img):
    """اكتشاف الشبكة عبر خطوط Hough كخطة بديلة"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        if lines is None:
            return None
        horizontal, vertical = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if angle < 15 and length > 80:
                horizontal.append(line[0])
            elif angle > 75 and length > 80:
                vertical.append(line[0])
        if len(horizontal) < 2 or len(vertical) < 2:
            return None
        h_arr = np.array(horizontal)
        v_arr = np.array(vertical)
        min_y = min(h_arr[:, 1].min(), h_arr[:, 3].min())
        max_y = max(h_arr[:, 1].max(), h_arr[:, 3].max())
        min_x = min(v_arr[:, 0].min(), v_arr[:, 2].min())
        max_x = max(v_arr[:, 0].max(), v_arr[:, 2].max())
        if (max_y - min_y) < 100 or (max_x - min_x) < 100:
            return None
        pts = np.array([
            [[min_x, min_y]],
            [[max_x, min_y]],
            [[max_x, max_y]],
            [[min_x, max_y]]
        ], dtype=np.float32)
        return pts
    except:
        return None

def find_board_robust(img):
    """محاولات متعددة بمعاملات مختلفة لإيجاد الشبكة"""
    params = [
        (2.0, 11, 2),
        (3.0, 11, 2),
        (2.0, 15, 3),
        (4.0, 11, 4),
        (2.0, 7, 2),
        (3.0, 15, 4),
        (5.0, 11, 2),
    ]
    for clip, block, c in params:
        thresh = preprocess_image(img, clip, block, c)
        pts = find_board(thresh)
        if pts is not None:
            return pts, thresh
    # خطة بديلة: Hough Lines
    pts = find_board_hough(img)
    if pts is not None:
        thresh = preprocess_image(img)
        return pts, thresh
    return None, None

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
# 2. استخراج الأرقام (تم التحديث لإزالة التشويش)
# ==========================================
def prepare_cell(cell):
    """استخراج الرقم من الخلية وتصفيته وتجهيزه بمقاس 28×28"""
    if cell is None or cell.size == 0:
        return None
        
    # 1. إزالة التشويش النقطي الصغير باستخدام Morphological Opening
    kernel = np.ones((2, 2), np.uint8)
    cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(
        cell,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
        
    valid_contours = []
    h_cell, w_cell = cell.shape
    
    # 2. فلترة الكنتورات بذكاء
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # استبعاد النقاط الصغيرة جداً (التشويش) والضخمة جداً
        if area < 40 or area > (h_cell * w_cell * 0.8):
            continue
            
        # استبعاد الكنتورات التي تلامس حواف الخلية (بقايا خطوط الشبكة)
        if x < 2 or y < 2 or (x + w) > w_cell - 2 or (y + h) > h_cell - 2:
            continue
            
        # التأكد من أن الأبعاد منطقية للرقم
        aspect_ratio = w / float(h)
        if 0.1 < aspect_ratio < 1.5: 
            valid_contours.append(c)

    if not valid_contours:
        return None
        
    # اختيار الكنتور الأكبر مساحة من بين الصالحة
    best_c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best_c)
    
    if w < 3 or h < 3:
        return None
        
    digit = cell[y:y + h, x:x + w]
    if digit.size == 0:
        return None
        
    # 3. توسيط الرقم في قماش 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    scale = 20.0 / max(w, h)
    nw = max(int(w * scale), 1)
    nh = max(int(h * scale), 1)
    res = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # تحسين وضوح الرقم بعد تغيير حجمه
    _, res = cv2.threshold(res, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    oy = (28 - nh) // 2
    ox = (28 - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = res
    return canvas

def get_cell_multi_threshold(gray_cell):
    """إنشاء نسخ متعددة بطرق threshold مختلفة"""
    cells = []
    try:
        _, t1 = cv2.threshold(
            gray_cell,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        cells.append(t1)
    except:
        pass
    try:
        t2 = cv2.adaptiveThreshold(
            gray_cell,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        cells.append(t2)
    except:
        pass
    try:
        t3 = cv2.adaptiveThreshold(
            gray_cell,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        cells.append(t3)
    except:
        pass
    return cells

def augment_canvas(canvas):
    """إنشاء نسخ معدّلة (TTA) من صورة 28×28"""
    versions = [canvas]
    # تدوير بسيط
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        rotated = cv2.warpAffine(canvas, M, (28, 28))
        versions.append(rotated)
    # إزاحة بسيطة
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        M_s = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(canvas, M_s, (28, 28))
        versions.append(shifted)
    return versions

def extract_digits_batch(warped_img, model, use_tta=True, conf_threshold=0.7):
    """استخراج الأرقام بدفعة واحدة"""
    board = np.zeros((9, 9), dtype=int)
    confidences = np.zeros((9, 9), dtype=float)
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    debug_montage = np.zeros((9 * 28, 9 * 28), dtype=np.uint8)
    cell_size = 450 // 9
    cell_data = {}
    progress = st.progress(0, text="🤖 تحليل الخلايا...")

    # ────── المرحلة 1: جمع جميع الخلايا ──────
    for i in range(9):
        for j in range(9):
            # تم التحديث هنا إلى 15% لتجاوز خطوط الشبكة السميكة
            m = int(cell_size * 0.15)
            y1, y2 = i * cell_size + m, (i + 1) * cell_size - m
            x1, x2 = j * cell_size + m, (j + 1) * cell_size - m
            gray_cell = gray[y1:y2, x1:x2]
            if gray_cell.size == 0:
                continue

            # Multi-threshold
            thresh_versions = get_cell_multi_threshold(gray_cell)
            canvases = []
            first_canvas = None
            for tv in thresh_versions:
                c = prepare_cell(tv)
                if c is not None:
                    if first_canvas is None:
                        first_canvas = c
                    canvases.append(c)

            if canvases and first_canvas is not None:
                debug_montage[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = first_canvas

                # TTA
                if use_tta:
                    augmented = augment_canvas(first_canvas)
                    canvases.extend(augmented[1:])

                cell_data[(i, j)] = canvases
            progress.progress(((i * 9 + j) + 1) / 81)
    progress.empty()

    # ────── المرحلة 2: تنبؤ بدفعة واحدة ⚡ ──────
    if cell_data:
        all_images = []
        cell_indices = []
        for (i, j), canvases in cell_data.items():
            for canvas in canvases:
                all_images.append(canvas)
                cell_indices.append((i, j))

        if all_images:
            batch = np.array(all_images).reshape(
                -1, 28, 28, 1
            ).astype('float32') / 255.0

            with st.spinner(f"⚡ تنبؤ دفعة واحدة ({len(all_images)} صورة)..."):
                predictions = model.predict(batch, verbose=0)

            # تجميع التنبؤات لكل خلية
            cell_preds = {}
            for idx, (i, j) in enumerate(cell_indices):
                if (i, j) not in cell_preds:
                    cell_preds[(i, j)] = []
                cell_preds[(i, j)].append(predictions[idx])

            for (i, j), preds in cell_preds.items():
                avg_pred = np.mean(preds, axis=0)
                digit = int(np.argmax(avg_pred))
                conf = float(avg_pred[digit])
                if conf > conf_threshold and digit != 0:
                    board[i][j] = digit
                    confidences[i][j] = conf

    st.session_state.debug_clean = debug_montage
    return board, confidences

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
                    return False, f"تكرار الرقم {v} في الموقع [صف {i+1}, عمود {j+1}]"
    return True, ""

# ==========================================
# 4. رسم الحل على الصورة
# ==========================================
def draw_solution_on_warped(warped_img, solved, original):
    result = warped_img.copy()
    h, w = result.shape[:2]
    ch, cw = h // 9, w // 9
    for i in range(9):
        for j in range(9):
            if original[i, j] == 0 and solved[i, j] != 0:
                txt = str(solved[i, j])
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs, thick = 1.2, 2
                sz = cv2.getTextSize(txt, font, fs, thick)[0]
                tx = j * cw + (cw - sz[0]) // 2
                ty = i * ch + (ch + sz[1]) // 2
                pad = 4
                cv2.rectangle(
                    result,
                    (tx - pad, ty - sz[1] - pad),
                    (tx + sz[0] + pad, ty + pad),
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
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:] = 255
    mask_warped = cv2.warpPerspective(mask, M_inv, (w, h))
    result = original_img.copy()
    mask_3ch = cv2.cvtColor(mask_warped, cv2.COLOR_GRAY2BGR)
    result = np.where(mask_3ch > 0, warped_back, result)
    return result

# ==========================================
# 5. مكونات الواجهة
# ==========================================
def show_confidence_board(board, confidences):
    """عرض اللوحة المكتشفة بألوان حسب الثقة"""
    html = """<table style='border-collapse:collapse; margin:auto; font-family:monospace;'>"""
    for i in range(9):
        html += "<tr>"
        for j in range(9):
            val = board[i][j]
            conf = confidences[i][j]
            if val == 0:
                bg = "#f5f5f5"
                text = ""
            elif conf > 0.95:
                bg = "#c8e6c9"
                text = str(val)
            elif conf > 0.80:
                bg = "#fff9c4"
                text = str(val)
            else:
                bg = "#ffcdd2"
                text = f"{val}?"
            # حدود المربعات 3×3
            bt = "3px solid #000" if i % 3 == 0 else "1px solid #aaa"
            bl = "3px solid #000" if j % 3 == 0 else "1px solid #aaa"
            bb = "3px solid #000" if i == 8 else ""
            br = "3px solid #000" if j == 8 else ""
            style = (
                f"width:46px; height:46px; text-align:center; "
                f"font-size:20px; font-weight:bold; background:{bg}; "
                f"border-top:{bt}; border-left:{bl};"
            )
            if bb:
                style += f" border-bottom:{bb};"
            if br:
                style += f" border-right:{br};"
            html += f"<td style='{style}'>{text}</td>"
        html += "</tr>"
    html += "</table>"
    html += """
    <div style='text-align:center; margin-top:12px; font-size:13px;'>
        <span style='background:#c8e6c9; padding:3px 10px; border-radius:4px; margin:0 3px;'>🟢 ثقة عالية &gt;95%</span>
        <span style='background:#fff9c4; padding:3px 10px; border-radius:4px; margin:0 3px;'>🟡 متوسطة &gt;80%</span>
        <span style='background:#ffcdd2; padding:3px 10px; border-radius:4px; margin:0 3px;'>🔴 منخفضة &lt;80%</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def get_download_button(img, filename="sudoku_solved.png"):
    """زر تحميل الصورة المحلولة"""
    success, buffer = cv2.imencode('.png', img)
    if success:
        st.download_button(
            label="📥 تحميل الحل كصورة PNG",
            data=BytesIO(buffer.tobytes()),
            file_name=filename,
            mime="image/png",
            use_container_width=True
        )

def save_history(original, solved):
    """حفظ اللغز في السجل"""
    st.session_state.history.append({
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'clues': int(np.count_nonzero(original)),
        'original': original.copy(),
        'solved': solved.copy(),
    })

def show_history():
    """عرض سجل الألغاز المحلولة"""
    if not st.session_state.history:
        st.caption("لا توجد ألغاز محلولة بعد")
        return
    for idx, entry in enumerate(reversed(st.session_state.history)):
        num = len(st.session_state.history) - idx
        with st.expander(
            f"🧩 لغز #{num} — {entry['time']} "
            f"({entry['clues']} أرقام)"
        ):
            sol_df = pd.DataFrame(
                entry['solved'],
                columns=[f"C{i+1}" for i in range(9)],
                index=[f"R{i+1}" for i in range(9)]
            )
            st.dataframe(sol_df, use_container_width=True)
    if st.button("🗑️ مسح السجل"):
        st.session_state.history = []
        st.rerun()

def reset_state():
    """إعادة تعيين الحالة للصورة الجديدة"""
    st.session_state.update({
        'img_hash': None,
        'board_extracted': False,
        'extracted_board': None,
        'confidences': None,
        'solved_img': None,
        'solved_board': None,
        'debug_clean': None,
        'warped_img': None,
        'original_img': None,
        'pts': None,
    })

# ==========================================
# ██ واجهة التطبيق الرئيسية ██
# ==========================================
st.title("🤖 حلّال السودوكو الذكي النهائي")
st.caption(
    "CNN + Multi-Threshold + TTA + Batch Prediction + Hough Fallback + Noise Filtering"
)

# ═══════════ الشريط الجانبي ═══════════
with st.sidebar:
    st.header("⚙️ الإعدادات")
    use_tta = st.checkbox(
        "🔄 Test-Time Augmentation",
        value=True,
        help="يزيد دقة التعرف عبر إنشاء نسخ معدّلة من كل خلية"
    )
    conf_threshold = st.slider(
        "🎯 حد الثقة الأدنى",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.05,
        help="الأرقام بثقة أقل من هذا الحد تُعتبر فارغة"
    )
    st.divider()
    st.header("📜 سجل الألغاز")
    show_history()

# ═══════════ تحميل النموذج ═══════════
model = load_digit_model()
if model is None:
    st.error("⚠️ ملف model.h5 مفقود أو تالف!")
    st.info(
        "ضع ملف `model.h5` (نموذج CNN مدرّب على MNIST) "
        "في نفس مجلد التطبيق."
    )
    st.stop()
st.success("✅ النموذج جاهز للعمل")

# ═══════════ اختيار طريقة الإدخال ═══════════
input_mode = st.radio(
    "📥 طريقة الإدخال:",
    ("📸 كاميرا", "📁 ملف صورة / PDF", "⌨️ إدخال يدوي"),
    horizontal=True
)

# ╔══════════════════════════════════════════╗
# ║ الوضع اليدوي ║
# ╚══════════════════════════════════════════╝
if input_mode == "⌨️ إدخال يدوي":
    st.subheader("⌨️ أدخل أرقام السودوكو يدوياً")
    st.info("ضع **0** في الخلايا الفارغة")
    empty_board = np.zeros((9, 9), dtype=int)
    df = pd.DataFrame(
        empty_board,
        columns=[f"C{i+1}" for i in range(9)],
        index=[f"R{i+1}" for i in range(9)]
    )
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        key="manual_input"
    )
    if st.button(
        "🚀 حل السودوكو",
        type="primary",
        use_container_width=True
    ):
        final_board = edited_df.to_numpy().astype(int)
        ok, msg = validate_board(final_board)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            clue_count = np.count_nonzero(final_board)
            if clue_count < 17:
                st.warning(
                    f"⚠️ عدد الأرقام المُعطاة ({clue_count}) قليل جداً. "
                    f"الحد الأدنى النظري 17."
                )
            s_board = final_board.copy()
            original_board = final_board.copy()
            with st.spinner("⏳ جاري الحل..."):
                start = time.time()
                if solve(s_board):
                    elapsed = time.time() - start
                    st.success(f"✅ تم الحل في {elapsed:.2f} ثانية!")
                    st.balloons()
                    sol_df = pd.DataFrame(
                        s_board,
                        columns=[f"C{i+1}" for i in range(9)],
                        index=[f"R{i+1}" for i in range(9)]
                    )
                    st.dataframe(sol_df, use_container_width=True)
                    save_history(original_board, s_board)
                else:
                    st.error(
                        "❌ لا يمكن حل هذا اللغز! "
                        "تأكد من صحة الأرقام."
                    )

# ╔══════════════════════════════════════════╗
# ║ وضع الكاميرا أو الملف ║
# ╚══════════════════════════════════════════╝
else:
    if input_mode == "📸 كاميرا":
        upload = st.camera_input("📸 صوّر لغز السودوكو")
    else:
        upload = st.file_uploader(
            "📁 ارفع صورة أو ملف PDF",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'pdf']
        )

    if upload:
        data = upload.getvalue()
        filename = getattr(upload, 'name', 'photo.jpg').lower()

        # ── قراءة الصورة ──
        if filename.endswith('.pdf'):
            try:
                doc = fitz.open(stream=data, filetype="pdf")
                pix = doc[0].get_pixmap(dpi=200)
                img = cv2.imdecode(
                    np.frombuffer(pix.tobytes("png"), np.uint8),
                    1
                )
                doc.close()
            except Exception as e:
                st.error(f"❌ خطأ في قراءة PDF: {e}")
                st.stop()
        else:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

        if img is None:
            st.error("❌ فشل في قراءة الصورة!")
            st.stop()

        # ── ضبط الحجم ──
        img = resize_if_needed(img)

        # ── عرض الصورة الأصلية ──
        st.subheader("📷 الصورة الأصلية")
        st.image(img, channels="BGR", use_container_width=True)

        # ── التحقق من تغيير الصورة ──
        h_val = hashlib.md5(img.tobytes()[:5000]).hexdigest()
        if st.session_state.img_hash != h_val:
            st.session_state.update({
                'img_hash': h_val,
                'board_extracted': False,
                'extracted_board': None,
                'confidences': None,
                'solved_img': None,
                'solved_board': None,
                'debug_clean': None,
                'warped_img': None,
                'original_img': img.copy(),
                'pts': None,
            })

        # ══════════════════════════════════════
        # اكتشاف الشبكة
        # ══════════════════════════════════════
        with st.spinner("🔍 البحث عن شبكة السودوكو..."):
            pts, thresh = find_board_robust(img)

        if pts is None:
            st.error("❌ لم يتم العثور على شبكة سودوكو!")
            st.info(
                "💡 نصائح:\n"
                "- تأكد أن الصورة تحتوي شبكة كاملة واضحة\n"
                "- حسّن الإضاءة وزاوية التصوير\n"
                "- جرّب الاقتراب أكثر من اللغز"
            )
            st.stop()

        st.session_state.pts = pts
        warped, M = warp_image(img, pts)
        st.session_state.warped_img = warped.copy()
        st.subheader("🔲 الشبكة المكتشفة")
        st.image(warped, channels="BGR", use_container_width=True)

        # ══════════════════════════════════════
        # استخراج الأرقام
        # ══════════════════════════════════════
        if not st.session_state.board_extracted:
            board, confidences = extract_digits_batch(
                warped,
                model,
                use_tta=use_tta,
                conf_threshold=conf_threshold
            )
            st.session_state.extracted_board = board.copy()
            st.session_state.confidences = confidences.copy()
            st.session_state.board_extracted = True

            detected = int(np.count_nonzero(board))
            avg_conf = (
                float(np.mean(confidences[confidences > 0]))
                if detected > 0 else 0
            )
            st.info(
                f"🔢 تم اكتشاف **{detected}** رقم | "
                f"متوسط الثقة **{avg_conf:.1%}**"
            )

        # ══════════════════════════════════════
        # عرض اللوحة المكتشفة
        # ══════════════════════════════════════
        if st.session_state.board_extracted:
            st.subheader("🔢 اللوحة المكتشفة (ملونة حسب الثقة)")
            show_confidence_board(
                st.session_state.extracted_board,
                st.session_state.confidences
            )
            st.markdown("---")

            b = st.session_state.extracted_board.copy()
            ok, msg = validate_board(b)
            if not ok:
                st.warning(f"⚠️ {msg}")
                st.info("👇 عدّل الأرقام الخاطئة في الجدول أدناه")

            # ── جدول قابل للتعديل ──
            st.subheader("✏️ تعديل الأرقام (اختياري)")
            df = pd.DataFrame(
                b,
                columns=[f"C{i+1}" for i in range(9)],
                index=[f"R{i+1}" for i in range(9)]
            )
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                key="board_editor"
            )

            # ── زر الحل ──
            if st.button(
                "🚀 حل السودوكو",
                type="primary",
                use_container_width=True
            ):
                final_board = edited_df.to_numpy().astype(int)
                ok2, msg2 = validate_board(final_board)
                if not ok2:
                    st.error(f"❌ {msg2}")
                else:
                    clue_count = np.count_nonzero(final_board)
                    if clue_count < 17:
                        st.warning(
                            f"⚠️ عدد الأرقام ({clue_count}) قليل جداً"
                        )
                    s_board = final_board.copy()
                    original_board = final_board.copy()
                    with st.spinner("⏳ جاري الحل..."):
                        start_t = time.time()
                        solved = solve(s_board)
                        elapsed_t = time.time() - start_t
                        if solved:
                            st.success(
                                f"✅ تم الحل بنجاح في {elapsed_t:.2f} ثانية!"
                            )
                            st.balloons()
                            st.session_state.solved_board = s_board.copy()

                            # رسم الحل
                            solved_warped = draw_solution_on_warped(
                                st.session_state.warped_img.copy(),
                                s_board,
                                original_board
                            )
                            result = overlay_solution_on_original(
                                img,
                                solved_warped,
                                st.session_state.pts
                            )
                            st.session_state.solved_img = result
                            save_history(original_board, s_board)
                        else:
                            st.error(
                                "❌ لا يمكن حل هذا اللغز! "
                                "تأكد من صحة الأرقام المدخلة."
                            )

            # ══════════════════════════════════════
            # عرض النتيجة النهائية
            # ══════════════════════════════════════
            if st.session_state.solved_img is not None:
                st.markdown("---")
                st.subheader("🎯 النتيجة النهائية")

                # مقارنة قبل / بعد
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        img,
                        channels="BGR",
                        caption="📷 قبل الحل",
                        use_container_width=True
                    )
                with col2:
                    st.image(
                        st.session_state.solved_img,
                        channels="BGR",
                        caption="✅ بعد الحل",
                        use_container_width=True
                    )

                # زر التحميل
                get_download_button(st.session_state.solved_img)

                # الحل كجدول
                if st.session_state.solved_board is not None:
                    with st.expander("📊 عرض الحل كجدول"):
                        sol_df = pd.DataFrame(
                            st.session_state.solved_board,
                            columns=[f"C{i+1}" for i in range(9)],
                            index=[f"R{i+1}" for i in range(9)]
                        )
                        st.dataframe(sol_df, use_container_width=True)

            # ══════════════════════════════════════
            # المعاينة التقنية
            # ══════════════════════════════════════
            if st.session_state.debug_clean is not None:
                with st.expander("🛠️ المعاينة التقنية (Debug) - مصفاة من التشويش"):
                    st.image(
                        st.session_state.debug_clean,
                        caption="الأرقام الصافية كما رآها الذكاء الاصطناعي (28×28)",
                        use_container_width=True
                    )
                    if st.session_state.confidences is not None:
                        conf = st.session_state.confidences
                        non_zero = conf[conf > 0]
                        if len(non_zero) > 0:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("أعلى ثقة", f"{non_zero.max():.1%}")
                            c2.metric("أقل ثقة", f"{non_zero.min():.1%}")
                            c3.metric("المتوسط", f"{non_zero.mean():.1%}")
