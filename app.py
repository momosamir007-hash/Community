import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd
import fitz  # مكتبة PyMuPDF للتعامل مع ملفات PDF

# ==========================================
# إعدادات الصفحة (يجب أن يكون أول أمر Streamlit)
# ==========================================
st.set_page_config(page_title="Sudoku AR Solver", page_icon="🧩", layout="centered")

# ==========================================
# إعدادات الـ API (Cloud OCR) - باستخدام Secrets
# ==========================================
try:
    API_KEY = st.secrets["OCR_API_KEY"]
except KeyError:
    st.error("⚠️ لم يتم العثور على مفتاح API. الرجاء إعداد secrets.toml")
    st.stop()

OCR_URL = 'https://api.ocr.space/parse/image'

# ==========================================
# ❶ تهيئة session_state مبكراً لتجنب أخطاء KeyError
# ==========================================
# 🔴 خطأ سابق: لم يتم تهيئة المتغيرات مبكراً
# مما قد يسبب KeyError عند الوصول إليها قبل إنشائها
if 'current_img_hash' not in st.session_state:
    st.session_state.current_img_hash = None
if 'board_extracted' not in st.session_state:
    st.session_state.board_extracted = False
if 'original_board' not in st.session_state:
    st.session_state.original_board = None
if 'solved_result_img' not in st.session_state:
    st.session_state.solved_result_img = None
if 'solved_board' not in st.session_state:
    st.session_state.solved_board = None

# ==========================================
# 1. دوال معالجة الصور (OpenCV)
# ==========================================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    # 🔴 خطأ سابق: استخدام أرقام بدلاً من ثوابت OpenCV
    # كان: cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    # المشكلة: صعوبة القراءة واحتمال الخطأ في القيم
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh

def find_board(thresh_img):
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_area = 0
    biggest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > largest_area and len(approx) == 4:
                largest_area = area
                biggest_contour = approx
    return biggest_contour

def order_points(pts):
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_image(img, pts, width=450, height=450):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, matrix, (width, height))
    return warped, matrix

# ==========================================
# 2. التواصل مع الـ API و OCR
# ==========================================
def extract_digits_via_api(warped_img):
    _, buffer = cv2.imencode('.jpg', warped_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    base64_payload = f"data:image/jpeg;base64,{img_base64}"

    payload = {
        'apikey': API_KEY,
        'base64Image': base64_payload,
        'OCREngine': '2',
        'isTable': 'true',
        'scale': 'true'
    }

    try:
        response = requests.post(OCR_URL, data=payload, timeout=20)
        result = response.json()
        if result.get('IsErroredOnProcessing'):
            error_msg = result.get('ErrorMessage', 'خطأ غير معروف')
            st.error(f"خطأ من الخادم: {error_msg}")
            return None

        # 🔴 خطأ سابق: لم يكن هناك تحقق من وجود ParsedResults
        # إذا كانت القائمة فارغة أو غير موجودة يحدث crash
        parsed_results = result.get('ParsedResults')
        if not parsed_results or len(parsed_results) == 0:
            st.error("❌ لم يتم الحصول على نتائج من الـ OCR.")
            return None

        parsed_text = parsed_results[0].get('ParsedText', '')
        if not parsed_text.strip():
            st.warning("⚠️ لم يتم التعرف على أي نص في الصورة.")
            return None

        return parse_api_text_to_grid(parsed_text)

    except requests.exceptions.Timeout:
        st.error("⏰ انتهت مهلة الاتصال بالخادم. حاول مرة أخرى.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 فشل الاتصال بالإنترنت.")
        return None
    except Exception as e:
        st.error(f"فشل الاتصال بالـ API: {e}")
        return None

def parse_api_text_to_grid(text):
    board = np.zeros((9, 9), dtype=int)
    lines = text.split('\n')
    row_idx = 0
    for line in lines:
        if not line.strip():
            continue
        # 🔴 خطأ سابق: استخدام split() ثم int(d)
        # المشكلة: إذا أعاد OCR "45" كرقم واحد، يصبح int("45") = 45
        # وهذا غير صالح في السودوكو (يجب أن يكون 1-9 أو 0)
        # الحل: استخراج كل حرف رقمي منفرداً
        digits = [int(c) for c in line if c.isdigit()]
        if len(digits) > 0 and row_idx < 9:
            for col_idx, val in enumerate(digits[:9]):
                # 🔴 خطأ سابق: لم يكن هناك تحقق من نطاق القيمة
                if 0 <= val <= 9:
                    board[row_idx][col_idx] = val
                else:
                    board[row_idx][col_idx] = 0
            row_idx += 1
    return board

# ==========================================
# 3. خوارزمية الحل (Backtracking)
# ==========================================
def is_valid(board, r, c, num):
    for i in range(9):
        if board[r][i] == num or board[i][c] == num:
            return False
    start_r, start_c = r - r % 3, c - c % 3
    for i in range(3):
        for j in range(3):
            if board[i + start_r][j + start_c] == num:
                return False
    return True

def solve_sudoku(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for num in range(1, 10):
                    if is_valid(board, r, c, num):
                        board[r][c] = num
                        if solve_sudoku(board):
                            return True
                        board[r][c] = 0
                return False
    return True

# ==========================================
# ❹ دالة التحقق من صحة الشبكة قبل الحل
# ==========================================
# 🔴 خطأ سابق: لم يكن هناك تحقق من صحة الأرقام المدخلة
# مما يسبب دخول الخوارزمية في حلقة لا نهائية أو نتائج خاطئة
def validate_board(board):
    """يتحقق من أن الشبكة لا تحتوي على تكرارات غير صالحة"""
    for i in range(9):
        for j in range(9):
            val = board[i][j]
            if val < 0 or val > 9:
                return False, f"القيمة {val} في الصف {i+1} العمود {j+1} خارج النطاق (0-9)"
            if val != 0:
                board[i][j] = 0  # أزل القيمة مؤقتاً للتحقق
                if not is_valid(board, i, j, val):
                    board[i][j] = val  # أرجعها
                    return False, f"الرقم {val} مكرر في الصف {i+1} أو العمود {j+1} أو المربع"
                board[i][j] = val  # أرجعها
    return True, ""

# ==========================================
# 4. الواقع المعزز (AR Overlay)
# ==========================================
def display_numbers(img, solved_board, original_board):
    cell_w = img.shape[1] // 9
    cell_h = img.shape[0] // 9
    for i in range(9):
        for j in range(9):
            if original_board[i][j] == 0 and solved_board[i][j] != 0:
                text = str(solved_board[i][j])
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )[0]
                text_x = (j * cell_w) + (cell_w - text_size[0]) // 2
                text_y = (i * cell_h) + (cell_h + text_size[1]) // 2
                cv2.putText(
                    img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
    return img

# ==========================================
# واجهة Streamlit التفاعلية
# ==========================================
st.title("🧩 حلّال السودوكو الذكي بالواقع المعزز")
st.write("التقط صورة لشبكة السودوكو، أو قم برفع ملف (صورة/PDF)، وسيقوم التطبيق بتحليلها وحلها.")

input_method = st.radio(
    "اختر طريقة إدخال اللغز:",
    ("📸 استخدام الكاميرا", "📁 رفع ملف (صورة أو PDF)"),
    key="input_method_radio"  # 🔴 إضافة key لتجنب تعارض الحالات
)

cv2_img = None

if input_method == "📸 استخدام الكاميرا":
    camera_image = st.camera_input(
        "وجه الكاميرا نحو السودوكو والتقط صورة",
        key="camera_input_widget"
    )
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(
            np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
        )

elif input_method == "📁 رفع ملف (صورة أو PDF)":
    uploaded_file = st.file_uploader(
        "قم بسحب وإفلات الملف هنا",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        key="file_uploader_widget"
    )
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith('.pdf'):
            with st.spinner("جاري استخراج الصفحة من الـ PDF..."):
                try:
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    # 🔴 خطأ سابق: لم يكن هناك تحقق من أن الـ PDF يحتوي صفحات
                    if doc.page_count == 0:
                        st.error("❌ ملف PDF فارغ!")
                    else:
                        page = doc.load_page(0)
                        pix = page.get_pixmap(dpi=150)
                        img_data = pix.tobytes("png")
                        cv2_img = cv2.imdecode(
                            np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR
                        )
                    doc.close()  # 🔴 إضافة: إغلاق الملف بعد الانتهاء
                except Exception as e:
                    st.error(f"❌ خطأ في قراءة ملف PDF: {e}")
        else:
            bytes_data = uploaded_file.read()
            cv2_img = cv2.imdecode(
                np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
            )

# ==========================================
# معالجة الصورة وعرض النتائج
# ==========================================
if cv2_img is not None:
    # 🔴 خطأ سابق: hash() على بيانات كبيرة قد يكون بطيئاً وغير موثوق
    # الحل: استخدام hashlib أو عينة من البيانات
    img_bytes = cv2_img.tobytes()
    img_hash = hash(img_bytes[:10000])  # عينة فقط للسرعة

    if st.session_state.current_img_hash != img_hash:
        st.session_state.current_img_hash = img_hash
        st.session_state.board_extracted = False
        st.session_state.original_board = None
        st.session_state.solved_result_img = None
        st.session_state.solved_board = None

    thresh = preprocess_image(cv2_img)
    contour = find_board(thresh)

    if contour is not None:
        width, height = 450, 450
        warped, matrix = warp_image(cv2_img, contour, width, height)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(
                warped, channels="BGR",
                caption="الشبكة المستخرجة (تم تصحيح المنظور)",
                use_container_width=True
            )

        # استخراج الأرقام مرة واحدة فقط
        if not st.session_state.board_extracted:
            with st.spinner("⏳ جاري قراءة الأرقام عبر الـ API..."):
                board = extract_digits_via_api(warped)
                if board is not None:
                    st.session_state.original_board = board.copy()
                    st.session_state.board_extracted = True

        with col2:
            if (st.session_state.board_extracted and
                    st.session_state.original_board is not None):
                st.markdown("### 📝 مراجعة الأرقام")
                st.info("الرقم (0) = خلية فارغة. عدّل أي خلية خاطئة.")
                df = pd.DataFrame(
                    st.session_state.original_board,
                    columns=[str(i) for i in range(1, 10)]
                )
                edited_df = st.data_editor(
                    df, hide_index=True, use_container_width=True,
                    key="sudoku_editor"  # 🔴 إضافة key مهمة
                )

                if st.button(
                    "🚀 تأكيد وحل اللغز",
                    type="primary", use_container_width=True,
                    key="solve_button"
                ):
                    # 🔴 خطأ سابق: لم يكن هناك .astype(int)
                    # مما قد يسبب أخطاء إذا أدخل المستخدم قيماً عشرية
                    try:
                        corrected_board = edited_df.to_numpy().astype(int)
                    except (ValueError, TypeError):
                        st.error("❌ الرجاء إدخال أرقام صحيحة فقط (0-9)")
                        st.stop()

                    # 🔴 إضافة: تحقق من صحة الشبكة قبل الحل
                    is_ok, error_msg = validate_board(corrected_board.copy())
                    if not is_ok:
                        st.error(f"⚠️ الشبكة غير صالحة: {error_msg}")
                        st.stop()

                    solved_board = corrected_board.copy()
                    with st.spinner("🧠 جاري الحل..."):
                        if solve_sudoku(solved_board):
                            st.success("✅ تم إيجاد الحل!")

                            # تركيب AR
                            ar_layer = np.zeros(
                                (height, width, 3), dtype=np.uint8
                            )
                            ar_layer = display_numbers(
                                ar_layer, solved_board, corrected_board
                            )

                            inverse_matrix = cv2.getPerspectiveTransform(
                                np.float32([
                                    [0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1],
                                    [0, height - 1]
                                ]),
                                order_points(contour)
                            )

                            inv_warp = cv2.warpPerspective(
                                ar_layer, inverse_matrix,
                                (cv2_img.shape[1], cv2_img.shape[0])
                            )

                            final_result = cv2.addWeighted(
                                cv2_img, 1, inv_warp, 1, 0
                            )

                            # 🔴 إضافة: حفظ النتيجة في session_state
                            # لمنع اختفائها عند إعادة التشغيل
                            st.session_state.solved_result_img = final_result
                            st.session_state.solved_board = solved_board
                            st.balloons()
                        else:
                            st.error(
                                "⚠️ هذه الشبكة غير قابلة للحل. "
                                "تأكد من عدم تكرار الأرقام بشكل خاطئ."
                            )

    else:
        st.error(
            "❌ لم يتم العثور على شبكة سودوكو واضحة. "
            "يرجى التأكد من أن الصورة تحتوي على مربع واضح المعالم."
        )

# 🔴 إضافة: عرض النتيجة المحفوظة حتى بعد إعادة التشغيل
if st.session_state.solved_result_img is not None:
    st.markdown("---")
    st.markdown("### 🎯 النتيجة النهائية (الواقع المعزز)")
    st.image(
        st.session_state.solved_result_img, channels="BGR",
        caption="النتيجة النهائية", use_container_width=True
    )
    # عرض الشبكة المحلولة كجدول أيضاً
    if st.session_state.solved_board is not None:
        st.markdown("### 📊 الحل كجدول")
        solved_df = pd.DataFrame(
            st.session_state.solved_board,
            columns=[str(i) for i in range(1, 10)],
            index=[str(i) for i in range(1, 10)]
        )
        st.dataframe(solved_df, use_container_width=True)
