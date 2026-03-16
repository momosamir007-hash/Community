import streamlit as st
import cv2
import numpy as np
import requests
import base64
import pandas as pd

# ==========================================
# إعدادات الـ API (Cloud OCR)
# ==========================================
API_KEY = st.secrets["OCR_API_KEY"]
OCR_URL = 'https://api.ocr.space/parse/image'

# ==========================================
# 1. دوال معالجة الصور (OpenCV)
# ==========================================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh

def find_board(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        [0, height - 1]], dtype="float32")
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
        response = requests.post(OCR_URL, data=payload, timeout=15)
        result = response.json()
        if result.get('IsErroredOnProcessing'):
            st.error(f"خطأ من الخادم: {result.get('ErrorMessage')}")
            return None
        
        parsed_text = result['ParsedResults'][0]['ParsedText']
        return parse_api_text_to_grid(parsed_text)
    except Exception as e:
        st.error(f"فشل الاتصال بالـ API: {e}")
        return None

def parse_api_text_to_grid(text):
    board = np.zeros((9, 9), dtype=int)
    lines = text.split('\n')
    row_idx = 0
    for line in lines:
        cleaned_line = ''.join([c for c in line if c.isdigit() or c.isspace()])
        digits = [int(d) for d in cleaned_line.split() if d.isdigit()]
        if len(digits) > 0 and row_idx < 9:
            for col_idx, val in enumerate(digits[:9]):
                board[row_idx][col_idx] = val
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
# 4. الواقع المعزز (AR)
# ==========================================
def display_numbers(img, solved_board, original_board):
    cell_w = img.shape[1] // 9
    cell_h = img.shape[0] // 9
    for i in range(9):
        for j in range(9):
            # نرسم الرقم فقط إذا كانت الخلية الأصلية فارغة وتم حلها
            if original_board[i][j] == 0 and solved_board[i][j] != 0:
                text = str(solved_board[i][j])
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (j * cell_w) + (cell_w - text_size[0]) // 2
                text_y = (i * cell_h) + (cell_h + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

# ==========================================
# واجهة Streamlit التفاعلية
# ==========================================
st.set_page_config(page_title="Sudoku AR Solver", page_icon="🧩", layout="centered")

st.title("🧩 حلّال السودوكو بالواقع المعزز")
st.write("التقط صورة لشبكة السودوكو، راجع الأرقام، وسيقوم التطبيق بحلها.")

camera_image = st.camera_input("وجه الكاميرا نحو السودوكو والتقط صورة")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # استخدام Hash للصورة لمعرفة ما إذا قام المستخدم بالتقاط صورة جديدة
    img_hash = hash(bytes_data)
    if 'current_img_hash' not in st.session_state or st.session_state.current_img_hash != img_hash:
        st.session_state.current_img_hash = img_hash
        st.session_state.board_extracted = False
        st.session_state.original_board = None

    thresh = preprocess_image(cv2_img)
    contour = find_board(thresh)

    if contour is not None:
        width, height = 450, 450
        warped, matrix = warp_image(cv2_img, contour, width, height)
        st.image(warped, channels="BGR", caption="الشبكة المستخرجة (للمقارنة)", use_container_width=True)
        
        # استخراج الأرقام مرة واحدة فقط لكل صورة جديدة
        if not st.session_state.board_extracted:
            with st.spinner("⏳ جاري قراءة الأرقام عبر الـ API..."):
                board = extract_digits_via_api(warped)
                if board is not None:
                    st.session_state.original_board = board
                    st.session_state.board_extracted = True
        
        # إذا تم استخراج الأرقام بنجاح، نعرض جدول المراجعة
        if st.session_state.board_extracted and st.session_state.original_board is not None:
            st.markdown("### 📝 مراجعة وتصحيح الأرقام")
            st.info("الرقم (0) يمثل الخلية الفارغة. اضغط مرتين على أي خلية لتعديلها إذا أخطأ الخادم في قراءتها.")
            
            # تحويل المصفوفة إلى DataFrame لسهولة التعديل في الواجهة
            df = pd.DataFrame(st.session_state.original_board, columns=[str(i) for i in range(1, 10)])
            
            # عرض الجدول التفاعلي القابل للتعديل
            edited_df = st.data_editor(df, hide_index=True, use_container_width=True)
            
            if st.button("🚀 تأكيد وحل اللغز", type="primary"):
                # تحويل الجدول المعدل إلى مصفوفة Numpy مجدداً
                corrected_board = edited_df.to_numpy()
                solved_board = corrected_board.copy()
                
                with st.spinner("🧠 جاري تشغيل خوارزمية الحل..."):
                    if solve_sudoku(solved_board):
                        st.success("✅ تم إيجاد الحل بنجاح!")
                        
                        # دمج الحل مع الصورة الأصلية (الواقع المعزز)
                        ar_layer = np.zeros((height, width, 3), dtype=np.uint8)
                        # نرسل corrected_board لكي لا يكتب التطبيق الأرقام فوق الأرقام التي صححتها أنت
                        ar_layer = display_numbers(ar_layer, solved_board, corrected_board)
                        
                        inverse_matrix = cv2.getPerspectiveTransform(
                            np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]), 
                            order_points(contour)
                        )
                        inv_warp = cv2.warpPerspective(ar_layer, inverse_matrix, (cv2_img.shape[1], cv2_img.shape[0]))
                        final_result = cv2.addWeighted(cv2_img, 1, inv_warp, 1, 0)
                        
                        st.image(final_result, channels="BGR", caption="النتيجة النهائية (الواقع المعزز)", use_container_width=True)
                        st.balloons() # تأثير احتفالي عند نجاح الحل!
                    else:
                        st.error("⚠️ الأرقام الموجودة في الجدول غير قابلة للحل. يرجى التأكد من عدم وجود أرقام مكررة في نفس الصف أو العمود.")
    else:
        st.error("❌ لم يتم العثور على شبكة سودوكو واضحة. يرجى التأكد من الإضاءة وإظهار حواف المربع بالكامل.")
