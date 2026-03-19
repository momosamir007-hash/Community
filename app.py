
# ══════════════════════════════════════════════════════════════
# SudokuSense AI — النسخة المُصحَّحة والمُحدَّثة
# ══════════════════════════════════════════════════════════════

import streamlit as st

# ⚠️ يجب أن يكون أول أمر Streamlit قبل أي st.error / st.stop
st.set_page_config(
    page_title="SudokuSense AI",
    page_icon="🧩",
    layout="centered",
    initial_sidebar_state="collapsed",
)

import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
import sys
import os
import copy
import zipfile
import tempfile
import time

# ─── مسارات المشروع ───
sys.path.append("./ImageProcess")
sys.path.append("./Results")
sys.path.append("./Solver")

# ─── TensorFlow ───
try:
    import tensorflow as tf

    TF_VERSION = tf.__version__
except ImportError:
    st.error("⚠️ TensorFlow غير مثبت! `pip install tensorflow`")
    st.stop()

# ─── وحدات المشروع (اختياري) ───
try:
    from processing import main_processing
except ImportError:
    main_processing = None

try:
    from solver import mainSolver
except ImportError:
    mainSolver = None

# ══════════════════════════════════════════════════════════════
# ثوابت عامة
# ══════════════════════════════════════════════════════════════
CONFUSED_PAIRS = {1: [7], 7: [1], 9: [4], 4: [9], 5: [2], 2: [5]}
GRID_SIZE = 9
CELL_PX = 50  # حجم الخلية في الصورة المقصوصة
WARP_SIZE = CELL_PX * GRID_SIZE  # 450
DEFAULTS = dict(
    grid=None,
    solved=None,
    original=None,
    confidence=None,
    alternatives=None,
    uncertain=set(),
    corrections=[],
    form_ver=0,  # عداد لتحديث مفاتيح النموذج
)


# ══════════════════════════════════════════════════════════════
# 1 · تحميل النموذج (6 طرق احتياطية)
# ══════════════════════════════════════════════════════════════
def _build_original_model():
    """يبني البنية الأصلية يدوياً (نفس أسماء الطبقات في الملف)."""
    m = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1), name="conv2d"
            ),
            tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv2d_1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d"),
            tf.keras.layers.Dropout(0.25, name="dropout"),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_2"),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_3"),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="max_pooling2d_1"),
            tf.keras.layers.Dropout(0.25, name="dropout_1"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="dense"),
            tf.keras.layers.Dropout(0.5, name="dropout_2"),
            tf.keras.layers.Dense(10, activation="softmax", name="dense_1"),
        ]
    )
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # بناء الطبقات
    m.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)
    return m


def _extract_h5_from_keras_zip(path):
    """ملف .keras = ZIP يحوي model.weights.h5 — نستخرجه."""
    try:
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".h5"):
                    out = os.path.join(tmp, "w.h5")
                    with open(out, "wb") as f:
                        f.write(zf.read(name))
                    return out
    except (zipfile.BadZipFile, Exception):
        pass
    return None


@st.cache_resource
def load_ai_model():
    """يحاول 6 طرق لتحميل النموذج بأي إصدار Keras."""
    # البحث عن الملف
    path = None
    for p in ("model.keras", "model.h5", "model_weights.weights.h5", "model_weights.h5"):
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return None, "لم يُعثر على أي ملف نموذج"

    errors = []

    # --- الطريقة 1 : تحميل مباشر ---
    try:
        return tf.keras.models.load_model(path), "تحميل مباشر ✅"
    except Exception as e:
        errors.append(f"1) {e}")

    # --- الطريقة 2 : compile=False ---
    try:
        m = tf.keras.models.load_model(path, compile=False)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m, "compile=False ✅"
    except Exception as e:
        errors.append(f"2) {e}")

    # --- الطريقة 3 : safe_mode=False ---
    try:
        m = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m, "safe_mode=False ✅"
    except Exception as e:
        errors.append(f"3) {e}")

    # --- الطريقة 4 : بناء يدوي + load_weights ---
    try:
        m = _build_original_model()
        m.load_weights(path)
        return m, "بناء يدوي + أوزان ✅"
    except Exception as e:
        errors.append(f"4) {e}")

    # --- الطريقة 5 : استخراج H5 من ZIP ---
    if path.endswith(".keras"):
        w = _extract_h5_from_keras_zip(path)
        if w:
            try:
                m = _build_original_model()
                m.load_weights(w)
                try:
                    os.remove(w)
                except OSError:
                    pass
                return m, "استخراج من ZIP ✅"
            except Exception as e:
                errors.append(f"5) {e}")

    # --- الطريقة 6 : ملفات أوزان منفصلة ---
    for wf in ("model_weights.weights.h5", "model_weights.h5", "weights.h5"):
        if os.path.exists(wf) and wf != path:
            try:
                m = _build_original_model()
                m.load_weights(wf)
                return m, f"أوزان منفصلة ({wf}) ✅"
            except Exception as e:
                errors.append(f"6-{wf}) {e}")

    return None, "\n".join(errors)


# ══════════════════════════════════════════════════════════════
# 2 · معالجة الصورة واستخراج الشبكة
# ══════════════════════════════════════════════════════════════
def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4,
    )


def find_grid_contour(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(
        [c for c in cnts if cv2.contourArea(c) > 1000],
        key=cv2.contourArea,
        reverse=True,
    )
    for cnt in cnts:
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def _order_points(pts):
    """ترتيب 4 نقاط: أعلى-يسار، أعلى-يمين، أسفل-يمين، أسفل-يسار."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def split_grid_into_cells(binary, contour):
    """
    بديل احتياطي لـ main_processing: يقص الشبكة بـ Perspective Transform
    ثم يقسمها لـ 81 خلية.
    """
    pts = _order_points(contour.astype(np.float32))
    dst = np.array(
        [[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(binary, M, (WARP_SIZE, WARP_SIZE))
    cells = []
    for row in range(9):
        for col in range(9):
            y1 = row * CELL_PX
            x1 = col * CELL_PX
            cell = warped[y1 : y1 + CELL_PX, x1 : x1 + CELL_PX]
            # هل الخلية تحتوي رقماً؟
            # قص الهوامش أولاً
            m = int(CELL_PX * 0.15)
            inner = cell[m : CELL_PX - m, m : CELL_PX - m]
            has_digit = np.sum(inner > 0) > (inner.size * 0.04)
            cells.append([cell, col, row, has_digit])
    return True, cells, warped


# ══════════════════════════════════════════════════════════════
# 3 · معالجة خلية الرقم الواحد
# ══════════════════════════════════════════════════════════════
def _preprocess_cell(cell_img):
    """
    يعيد قائمة نسخ (28×28 float32) جاهزة للنموذج.
    - CLAHE + عتبتان مختلفتان
    - قلب تلقائي للألوان
    - حفظ نسبة الأبعاد
    - توسيط بمركز الكتلة
    """
    if cell_img is None or cell_img.size == 0:
        return []
    gray = (
        cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        if len(cell_img.shape) == 3
        else cell_img.copy()
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # عتبتان
    _, t_otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    t_adapt = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    versions = []
    for thresh in (t_otsu, t_adapt):
        # قلب تلقائي
        if cv2.countNonZero(thresh) > thresh.size // 2:
            thresh = cv2.bitwise_not(thresh)

        h, w = thresh.shape
        mg = int(min(h, w) * 0.15)
        crop = thresh[mg : h - mg, mg : w - mg]

        # إزالة الغبار
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 25:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 3 or bh < 5:
            continue

        digit = crop[y : y + bh, x : x + bw]

        # حفظ نسبة الأبعاد → تصغير أطول ضلع إلى 20
        scale = 20.0 / max(bw, bh)
        nw = max(int(bw * scale), 1)
        nh = max(int(bh * scale), 1)
        resized = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

        # لوحة 28×28 مع التوسيط بمركز الكتلة
        canvas = np.zeros((28, 28), dtype=np.float32)

        # مركز الكتلة
        moments = cv2.moments(resized)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = nw // 2, nh // 2

        ox = max(14 - cx, 0)
        oy = max(14 - cy, 0)

        # تأكد من عدم الخروج عن الحدود
        end_x = min(ox + nw, 28)
        end_y = min(oy + nh, 28)
        src_w = end_x - ox
        src_h = end_y - oy

        canvas[oy:end_y, ox:end_x] = (
            resized[:src_h, :src_w].astype(np.float32) / 255.0
        )
        versions.append(canvas)

    return versions


# ══════════════════════════════════════════════════════════════
# 4 · التعرف مع الثقة + TTA
# ══════════════════════════════════════════════════════════════
def recognize_with_confidence(images, model, threshold=0.75):
    grid = [[0] * 9 for _ in range(9)]
    conf = [[1.0] * 9 for _ in range(9)]
    alts = [[[] for _ in range(9)] for _ in range(9)]
    uncertain = set()

    for item in images:
        cell_img, cx, cy, has = item[0], item[1], item[2], item[3]
        if not has:
            continue

        cell_versions = _preprocess_cell(cell_img)
        if not cell_versions:
            continue

        # TTA: إزاحة ±1 بكسل أفقياً ورأسياً
        augmented = []
        for v in cell_versions:
            augmented.append(v)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                augmented.append(cv2.warpAffine(v, M, (28, 28), borderValue=0))

        batch = np.array(augmented).reshape(-1, 28, 28, 1)
        preds = model.predict(batch, verbose=0)
        avg = np.mean(preds, axis=0)
        top = [i for i in np.argsort(avg)[::-1] if i != 0][:5]
        if not top:
            continue

        pred = top[0]
        c_val = float(avg[pred])
        grid[cy][cx] = pred
        conf[cy][cx] = c_val
        alts[cy][cx] = [(int(i), float(avg[i])) for i in top]

        # فحص الأزواج المتشابهة
        if pred in CONFUSED_PAIRS:
            for cd in CONFUSED_PAIRS[pred]:
                if float(avg[cd]) > c_val * 0.35:
                    uncertain.add((cy, cx))
        if c_val < threshold:
            uncertain.add((cy, cx))

    return grid, conf, alts, uncertain


# ══════════════════════════════════════════════════════════════
# 5 · كشف التعارضات + التصحيح التلقائي
# ══════════════════════════════════════════════════════════════
def find_conflicts(grid):
    bad = set()
    for i in range(9):
        # صف
        seen = {}
        for j in range(9):
            v = grid[i][j]
            if v:
                if v in seen:
                    bad.add((i, j))
                    bad.add((i, seen[v]))
                seen[v] = j
        # عمود
        seen = {}
        for j in range(9):
            v = grid[j][i]
            if v:
                if v in seen:
                    bad.add((j, i))
                    bad.add((seen[v], i))
                seen[v] = j
        # مربعات 3×3
        for br in range(3):
            for bc in range(3):
                seen = {}
                for dr in range(3):
                    for dc in range(3):
                        r, c = br * 3 + dr, bc * 3 + dc
                        v = grid[r][c]
                        if v:
                            if v in seen:
                                bad.add((r, c))
                                bad.add(seen[v])
                            seen[v] = (r, c)
    return bad


def auto_correct(grid, confidence, alternatives):
    g = copy.deepcopy(grid)
    cf = copy.deepcopy(confidence)
    fixes = []
    for _ in range(40):
        conflicts = find_conflicts(g)
        if not conflicts:
            break

        # الأضعف ثقة
        worst, wc = None, 999.0
        for pos in conflicts:
            if cf[pos[0]][pos[1]] < wc:
                wc = cf[pos[0]][pos[1]]
                worst = pos
        if worst is None:
            break

        r, c = worst
        old = g[r][c]
        fixed = False
        for av, ac in alternatives[r][c]:
            if av == old or av == 0:
                continue
            g[r][c] = av
            if len(find_conflicts(g)) < len(conflicts):
                fixes.append(
                    dict(
                        row=r,
                        col=c,
                        old=old,
                        new=av,
                        old_conf=wc,
                        new_conf=ac,
                    )
                )
                cf[r][c] = ac
                fixed = True
                break
        if not fixed:
            cf[r][c] = 999
    return g, fixes


# ══════════════════════════════════════════════════════════════
# 6 · حل السودوكو
# ══════════════════════════════════════════════════════════════
def _ok(g, r, c, n):
    if n in g[r]:
        return False
    if any(g[i][c] == n for i in range(9)):
        return False
    sr, sc = 3 * (r // 3), 3 * (c // 3)
    for i in range(sr, sr + 3):
        for j in range(sc, sc + 3):
            if g[i][j] == n:
                return False
    return True


def _bt(g):
    for i in range(9):
        for j in range(9):
            if g[i][j] == 0:
                for n in range(1, 10):
                    if _ok(g, i, j, n):
                        g[i][j] = n
                        if _bt(g):
                            return True
                        g[i][j] = 0
                return False
    return True


def solve_puzzle(grid):
    # المحلّل الأصلي أولاً
    if mainSolver is not None:
        try:
            res = mainSolver(copy.deepcopy(grid))
            if isinstance(res, list) and res != -1:
                return res
        except Exception:
            pass
    # Backtracking
    g = copy.deepcopy(grid)
    return g if _bt(g) else None


# ══════════════════════════════════════════════════════════════
# 7 · رسم الشبكة (صورة OpenCV + HTML)
# ══════════════════════════════════════════════════════════════
def draw_sudoku_image(grid, original=None):
    C = 64
    S = C * 9
    img = np.ones((S, S, 3), dtype=np.uint8) * 255

    for br in range(3):
        for bc in range(3):
            if (br + bc) % 2 == 0:
                y0, x0 = br * 3 * C, bc * 3 * C
                cv2.rectangle(
                    img,
                    (x0, y0),
                    (x0 + 3 * C, y0 + 3 * C),
                    (235, 232, 248),
                    -1,
                )

    for i in range(10):
        t = 4 if i % 3 == 0 else 1
        cl = (26, 35, 126) if i % 3 == 0 else (180, 180, 180)
        p = i * C
        cv2.line(img, (0, p), (S, p), cl, t)
        cv2.line(img, (p, 0), (p, S), cl, t)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(9):
        for x in range(9):
            v = int(grid[y][x])
            if v:
                orig = original and int(original[y][x]) != 0
                color = (126, 35, 26) if orig else (30, 120, 30)
                txt = str(v)
                ts = cv2.getTextSize(txt, font, 1.4, 2)[0]
                cv2.putText(
                    img,
                    txt,
                    (
                        x * C + (C - ts[0]) // 2,
                        y * C + (C + ts[1]) // 2,
                    ),
                    font,
                    1.4,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return img


def render_html(grid, original=None, uncertain=None, corrected=None, confidence=None, title=""):
    uncertain = uncertain or set()
    corrected = corrected or set()
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    .sw{display:flex;flex-direction:column;align-items:center; font-family:'Poppins',sans-serif;padding:10px}
    .stitle{font-size:18px;font-weight:700;color:#1a237e;margin-bottom:12px}
    .sb{border-collapse:collapse;border:4px solid #1a237e; border-radius:8px;overflow:hidden; box-shadow:0 8px 32px rgba(26,35,126,.18);background:#fff}
    .sb td{width:52px;height:52px;text-align:center;vertical-align:middle; font-size:26px;font-weight:700;border:1px solid #cfd8dc; transition:background .15s;position:relative;cursor:default}
    .sb td:hover{background:#fff9c4!important}
    .sb td.br{border-right:3px solid #1a237e!important}
    .sb td.bb{border-bottom:3px solid #1a237e!important}
    .sb td.bl{border-left:3px solid #1a237e!important}
    .sb td.bt{border-top:3px solid #1a237e!important}
    .sb td.orig{color:#1a237e;background:#e8eaf6}
    .sb td.solved{color:#1b5e20;background:#e8f5e9}
    .sb td.empty{color:#e0e0e0;background:#fafafa}
    .sb td.unc{color:#e65100!important;background:#fff3e0!important; animation:pulse 1.5s infinite}
    .sb td.fix{color:#6a1b9a!important;background:#f3e5f5!important}
    .sb td.shd{background:#f3f0ff}
    .sb td.shd.orig{background:#e0dcf7}
    .sb td.shd.solved{background:#d7f0d9}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
    .sb td .cf{position:absolute;bottom:1px;right:2px; font-size:7px;color:#999;font-weight:400}
    .lg{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;font-size:11px;color:#555}
    .li{display:flex;align-items:center;gap:4px}
    .lb{width:16px;height:16px;border-radius:3px;border:1px solid #bbb}
    </style>
    """
    h = css + '<div class="sw">'
    if title:
        h += f'<div class="stitle">{title}</div>'
    h += '<table class="sb">'
    for r in range(9):
        h += "<tr>"
        for c in range(9):
            v = int(grid[r][c])
            cl = []
            if c % 3 == 0 and c:
                cl.append("bl")
            if c % 3 == 2 and c < 8:
                cl.append("br")
            if r % 3 == 0 and r:
                cl.append("bt")
            if r % 3 == 2 and r < 8:
                cl.append("bb")
            if (r // 3 + c // 3) % 2 == 0:
                cl.append("shd")
            if (r, c) in corrected:
                cl.append("fix")
            elif (r, c) in uncertain:
                cl.append("unc")
            elif v == 0:
                cl.append("empty")
            elif original and int(original[r][c]):
                cl.append("orig")
            else:
                cl.append("solved")

            txt = str(v) if v else ""
            cf = ""
            if confidence and v and confidence[r][c] < 1.5:
                cf = f'<span class="cf">{confidence[r][c]:.0%}</span>'
            h += f'<td class="{" ".join(cl)}">{txt}{cf}</td>'
        h += "</tr>"
    h += "</table>"
    h += """
    <div class="lg">
        <div class="li"><div class="lb" style="background:#e8eaf6"></div> <b style="color:#1a237e">أصلي</b></div>
        <div class="li"><div class="lb" style="background:#e8f5e9"></div> <b style="color:#1b5e20">محلول</b></div>
        <div class="li"><div class="lb" style="background:#fff3e0"></div> <b style="color:#e65100">⚠️ مشكوك</b></div>
        <div class="li"><div class="lb" style="background:#f3e5f5"></div> <b style="color:#6a1b9a">🔧 مُصحّح</b></div>
    </div></div>"""
    return h


# ══════════════════════════════════════════════════════════════
# 8 · أدوات مساعدة
# ══════════════════════════════════════════════════════════════
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def _clear_form_keys():
    """حذف مفاتيح حقول الشبكة من session لتحديث القيم."""
    for r in range(9):
        for c in range(9):
            k = f"sc_{r}_{c}"
            if k in st.session_state:
                del st.session_state[k]


def _init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            # النسخ لتجنب المراجع المشتركة
            if isinstance(v, (list, dict, set)):
                st.session_state[k] = copy.deepcopy(v)
            else:
                st.session_state[k] = v


def _reset_state(**overrides):
    for k, v in DEFAULTS.items():
        if isinstance(v, (list, dict, set)):
            st.session_state[k] = copy.deepcopy(v)
        else:
            st.session_state[k] = v
    for k, v in overrides.items():
        st.session_state[k] = v
    _clear_form_keys()


# ══════════════════════════════════════════════════════════════
# 9 · الواجهة الرئيسية
# ══════════════════════════════════════════════════════════════

# ─── CSS عام ───
st.markdown(
    """
<style>
.block-container{max-width:820px}
div[data-testid="stForm"]{
    border:4px solid #1a237e!important;
    border-radius:14px!important;
    padding:22px 18px!important;
    background:linear-gradient(145deg,#f8f9ff,#eef0ff)!important;
    box-shadow:0 6px 24px rgba(26,35,126,.12)!important}
div[data-testid="stForm"] input[type="text"]{
    text-align:center!important;font-size:26px!important;
    font-weight:800!important;color:#1a237e!important;
    height:54px!important;border:1.5px solid #b0bec5!important;
    border-radius:6px!important;padding:0!important;
    background:#fff!important;transition:all .15s!important}
div[data-testid="stForm"] input[type="text"]:focus{
    border-color:#1565c0!important;
    box-shadow:0 0 0 3px rgba(21,101,192,.25)!important;
    background:#e3f2fd!important}
div[data-testid="stForm"] input[type="text"]::placeholder{
    color:#d0d0d0!important;font-size:20px!important}
.block-divider{border:none;height:4px;
    background:linear-gradient(90deg,transparent,#1a237e,transparent);
    margin:5px 0 7px 0;border-radius:4px}
</style>""",
    unsafe_allow_html=True,
)

# ─── عنوان ───
st.markdown(
    """
<div style="text-align:center;padding:10px 0 5px">
    <h1 style="color:#1a237e;margin:0">🧩 SudokuSense AI</h1>
    <p style="color:#666;font-size:15px;margin-top:6px">
        يستخرج ← يتحقق ← يُصحّح تلقائياً ← يحل
    </p>
</div>""",
    unsafe_allow_html=True,
)

_init_state()

# ─── تحميل النموذج ───
model, load_msg = load_ai_model()

# شريط جانبي للمعلومات
with st.sidebar:
    st.markdown("### ⚙️ معلومات النظام")
    st.write(f"**TensorFlow:** `{TF_VERSION}`")
    st.write(f"**النموذج:** {'✅ جاهز' if model else '❌ غير متاح'}")
    if model:
        st.write(f"**طريقة التحميل:** {load_msg}")
    else:
        with st.expander("تفاصيل الخطأ"):
            st.code(load_msg)
    st.write(f"**processing:** {'✅' if main_processing else '⛔ بديل داخلي'}")
    st.write(f"**solver:** {'✅' if mainSolver else '⛔ backtracking'}")
    st.markdown("---")
    st.markdown("### 📖 الألوان")
    st.markdown(
        """
    - 🔵 **أزرق** = رقم أصلي
    - 🟢 **أخضر** = رقم محلول
    - 🟠 **برتقالي** = مشكوك فيه
    - 🟣 **بنفسجي** = تصحيح تلقائي
    """
    )

# ═══════════════════════════════════════════
# التبويبات
# ═══════════════════════════════════════════
tab_img, tab_manual = st.tabs(["📷 استخراج من صورة", "✏️ إدخال يدوي"])

# ─── تبويب الصورة ───
with tab_img:
    uploaded = st.file_uploader(
        "ارفع صورة سودوكو",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        st.image(pil, caption="الصورة المرفوعة", width=320)

        if model is None:
            st.error("❌ النموذج غير متاح — راجع الشريط الجانبي")
        else:
            conf_thresh = st.slider(
                "🎚️ حد الثقة (كلما ↑ = فحص أدق)",
                0.50,
                0.95,
                0.75,
                0.05,
            )
            if st.button(
                "🔍 استخراج وتصحيح تلقائي",
                type="primary",
                use_container_width=True,
                key="btn_extract",
            ):
                cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

                with st.status("جاري المعالجة …", expanded=True) as status:
                    # 1 — ثنائية
                    st.write("🔍 معالجة الصورة …")
                    binary = binarize(cv_img)

                    # 2 — إيجاد الشبكة
                    contour = find_grid_contour(binary)
                    if contour is None:
                        status.update(label="❌ فشل", state="error")
                        st.error("لم يُعثر على شبكة سودوكو")
                        st.stop()

                    # 3 — تقسيم الخلايا
                    st.write("✂️ تقسيم الشبكة …")
                    if main_processing is not None:
                        try:
                            ok, imgs, _ = main_processing(contour, binary)
                        except Exception as e:
                            st.warning(
                                f"⚠️ main_processing فشل: {e} — سيُستخدم البديل الداخلي"
                            )
                            ok, imgs, _ = split_grid_into_cells(binary, contour)
                    else:
                        ok, imgs, _ = split_grid_into_cells(binary, contour)

                    if not ok:
                        status.update(label="❌ فشل", state="error")
                        st.error("فشل تقسيم الشبكة")
                        st.stop()

                    # 4 — قراءة الأرقام
                    st.write("🤖 قراءة الأرقام (TTA) …")
                    g, cf, al, unc = recognize_with_confidence(imgs, model, conf_thresh)

                    # 5 — تصحيح تلقائي
                    conflicts = find_conflicts(g)
                    st.write(f"🔎 تعارضات: **{len(conflicts)}**")
                    corr = []
                    if conflicts:
                        st.write("🔧 تصحيح تلقائي …")
                        g, corr = auto_correct(g, cf, al)
                        st.write(
                            f"✅ تصحيحات: **{len(corr)}** — متبقي: **{len(find_conflicts(g))}**"
                        )

                    # حفظ
                    st.session_state.grid = g
                    st.session_state.original = copy.deepcopy(g)
                    st.session_state.confidence = cf
                    st.session_state.alternatives = al
                    st.session_state.uncertain = unc
                    st.session_state.corrections = corr
                    st.session_state.solved = None
                    _clear_form_keys()

                    status.update(label="✅ اكتمل!", state="complete")

                if corr:
                    with st.expander("🔧 تفاصيل التصحيحات", expanded=True):
                        for f in corr:
                            st.write(
                                f"📍 ص{f['row']+1} ع{f['col']+1}: "
                                f"**{f['old']}** → **{f['new']}** "
                                f"({f['old_conf']:.0%} → {f['new_conf']:.0%})"
                            )
                safe_rerun()

# ─── تبويب الإدخال اليدوي ───
with tab_manual:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📝 شبكة فارغة", use_container_width=True, key="btn_empty"):
            empty = [[0] * 9 for _ in range(9)]
            _reset_state(grid=empty, original=copy.deepcopy(empty))
            safe_rerun()
    with c2:
        if st.button("🧪 مثال تجريبي", use_container_width=True, key="btn_example"):
            ex = [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
            _reset_state(grid=ex, original=copy.deepcopy(ex))
            safe_rerun()

    with st.expander("📋 لصق أرقام من نص"):
        txt = st.text_area(
            "9 أسطر — كل سطر 9 أرقام (0 = فارغ)",
            height=200,
            placeholder="530070000\n600195000\n098000060\n"
            "800060003\n400803001\n700020006\n"
            "060000280\n000419005\n000080079",
        )
        if st.button("⬇️ تحميل", key="btn_paste"):
            lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
            if len(lines) != 9:
                st.error("❌ يجب إدخال 9 أسطر بالضبط")
            else:
                g = []
                ok = True
                for line in lines:
                    ds = [
                        int(ch)
                        for ch in line.replace(" ", "").replace(",", "")
                        if ch.isdigit()
                    ]
                    if len(ds) != 9:
                        st.error(f"❌ «{line}» لا يحتوي 9 أرقام")
                        ok = False
                        break
                    g.append(ds)
                if ok:
                    _reset_state(grid=g, original=copy.deepcopy(g))
                    safe_rerun()

# ═══════════════════════════════════════════
# محرر الشبكة التفاعلي
# ═══════════════════════════════════════════
if st.session_state.grid is not None:
    st.markdown("---")

    # معاينة HTML
    corr_set = {(f["row"], f["col"]) for f in st.session_state.corrections}
    preview = render_html(
        st.session_state.grid,
        title="الشبكة الحالية",
        uncertain=st.session_state.uncertain,
        corrected=corr_set,
        confidence=st.session_state.confidence,
    )
    components.html(preview, height=560, scrolling=False)

    st.markdown(
        '<p style="text-align:center;color:#5c6bc0;font-weight:600;margin:10px 0 5px">'
        '⬇️ عدّل الأرقام ثم اضغط <b>🚀 حل</b></p>',
        unsafe_allow_html=True,
    )

    # ─── بناء النموذج (Form) ───
    edited = [[0] * 9 for _ in range(9)]
    action = None  # "solve" | "reset" | "clear"

    with st.form("sudoku_form", clear_on_submit=False):
        for brow in range(3):
            for lrow in range(3):
                row = brow * 3 + lrow
                # 9 خلايا + 2 فاصل رفيع
                widths = [1, 1, 1, 0.15, 1, 1, 1, 0.15, 1, 1, 1]
                cols = st.columns(widths, gap="small")
                idx_map = [0, 1, 2, 4, 5, 6, 8, 9, 10]
                for c in range(9):
                    with cols[idx_map[c]]:
                        v = st.session_state.grid[row][c]
                        disp = str(v) if v else ""
                        res = st.text_input(
                            label=f"R{row}C{c}",
                            value=disp,
                            key=f"sc_{row}_{c}",
                            max_chars=1,
                            label_visibility="collapsed",
                            placeholder="·",
                        )
                        if res and res.strip().isdigit():
                            d = int(res.strip())
                            edited[row][c] = d if 1 <= d <= 9 else 0
            if brow < 2:
                st.markdown('<hr class="block-divider">', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.form_submit_button("🚀 حل السودوكو", type="primary", use_container_width=True):
                action = "solve"
        with b2:
            if st.form_submit_button("🔄 إعادة تعيين", use_container_width=True):
                action = "reset"
        with b3:
            if st.form_submit_button("🗑️ مسح الكل", use_container_width=True):
                action = "clear"

    # ─── معالجة الأزرار (خارج الـ Form) ───
    if action == "solve":
        # التحقق
        valid = all(0 <= edited[r][c] <= 9 for r in range(9) for c in range(9))
        if not valid:
            st.error("❌ قيم غير صالحة (0-9 فقط)")
        else:
            st.session_state.grid = edited
            st.session_state.original = copy.deepcopy(edited)
            t0 = time.time()
            with st.spinner("⚙️ جاري الحل …"):
                result = solve_puzzle(edited)
                elapsed = time.time() - t0
            if result:
                st.session_state.solved = result
                _clear_form_keys()
                st.toast(f"⏱️ تم الحل في {elapsed:.2f} ثانية", icon="✅")
                safe_rerun()
            else:
                st.error("❌ الشبكة غير قابلة للحل! تحقق من الأرقام المدخلة.")
    elif action == "reset":
        st.session_state.grid = copy.deepcopy(
            st.session_state.original or [[0] * 9 for _ in range(9)]
        )
        st.session_state.solved = None
        _clear_form_keys()
        safe_rerun()
    elif action == "clear":
        _reset_state()
        safe_rerun()

# ═══════════════════════════════════════════
# عرض النتيجة النهائية
# ═══════════════════════════════════════════
if st.session_state.solved is not None:
    st.markdown("---")
    st.subheader("✅ الحل النهائي")
    solved_html = render_html(
        st.session_state.solved,
        st.session_state.original,
        title="🎉 تم الحل بنجاح!",
    )
    components.html(solved_html, height=580, scrolling=False)

    # تصدير كنص
    with st.expander("📊 عرض الحل كنص"):
        txt_lines = []
        for r in range(9):
            row_str = " ".join(str(st.session_state.solved[r][c]) for c in range(9))
            txt_lines.append(row_str)
            if r % 3 == 2 and r < 8:
                txt_lines.append("─" * 17)
        st.code("\n".join(txt_lines))

    # تحميل صورة
    result_img = draw_sudoku_image(st.session_state.solved, st.session_state.original)
    _, enc = cv2.imencode(".png", result_img)
    st.download_button(
        "📥 تحميل صورة الحل (PNG)",
        data=enc.tobytes(),
        file_name="sudoku_solved.png",
        mime="image/png",
        use_container_width=True,
    )

    st.balloons()
    st.success("🎉 تم حل السودوكو بنجاح!")
