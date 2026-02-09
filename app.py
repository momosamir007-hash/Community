import streamlit as st
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_DIRECTION
from cerebras.cloud.sdk import Cerebras
import json
import io

# ---------------------------------------------------------
# 1. إعداد التوقيت الأسبوعي
# ---------------------------------------------------------
WEEKLY_SCHEDULE = {
    "الأحد": [
        {"time": "08:00 - 09:45", "activity": "تعبير شفوي"},
        {"time": "08:00 - 09:45", "activity": "مبادئ القراءة"},
        {"time": "08:00 - 09:45", "activity": "رياضيات"},
        {"time": "10:00 - 11:15", "activity": "ت علمية"},
        {"time": "10:00 - 11:15", "activity": "ت إسلامية"},
        {"time": "13:00 - 15:00", "activity": "مسرح وعرائس"},
        {"time": "13:00 - 15:00", "activity": "رسم وأشغال"},
        {"time": "13:00 - 15:00", "activity": "ت بدنية"}
    ],
    "الاثنين": [
        {"time": "08:00 - 09:45", "activity": "رياضيات"},
        {"time": "08:00 - 09:45", "activity": "تعبير شفوي"},
        {"time": "08:00 - 09:45", "activity": "تخطيط"},
        {"time": "10:00 - 11:15", "activity": "ت علمية"},
        {"time": "10:00 - 11:15", "activity": "ت مدنية"},
        {"time": "13:00 - 15:00", "activity": "مسرح وعرائس"},
        {"time": "13:00 - 15:00", "activity": "رسم وأشغال"},
        {"time": "13:00 - 15:00", "activity": "ت بدنية"}
    ],
    "الثلاثاء": [
        {"time": "08:00 - 09:45", "activity": "تعبير شفوي"},
        {"time": "08:00 - 09:45", "activity": "مبادئ القراءة"},
        {"time": "08:00 - 09:45", "activity": "رياضيات"},
        {"time": "10:00 - 11:15", "activity": "ت إسلامية"},
        {"time": "10:00 - 11:15", "activity": "ت بدنية"}
    ],
    "الأربعاء": [
        {"time": "08:00 - 09:45", "activity": "رياضيات"},
        {"time": "08:00 - 09:45", "activity": "مبادئ القراءة"},
        {"time": "08:00 - 09:45", "activity": "تخطيط"},
        {"time": "10:00 - 11:15", "activity": "ت علمية"},
        {"time": "10:00 - 11:15", "activity": "ت مدنية"},
        {"time": "13:00 - 15:00", "activity": "ت إيقاعية"},
        {"time": "13:00 - 15:00", "activity": "موسيقى وإنشاد"},
        {"time": "13:00 - 15:00", "activity": "ت بدنية"}
    ],
    "الخميس": [
        {"time": "08:00 - 09:45", "activity": "مبادئ القراءة"},
        {"time": "08:00 - 09:45", "activity": "رياضيات"},
        {"time": "08:00 - 09:45", "activity": "ت علمية"},
        {"time": "10:00 - 11:15", "activity": "ت إيقاعية"},
        {"time": "10:00 - 11:15", "activity": "موسيقى وإنشاد"}
    ]
}

# ---------------------------------------------------------
# دالة مساعدة: تحديد المجال بناءً على النشاط
# ---------------------------------------------------------
def get_domain(activity):
    """تحدد المجال التربوي بناءً على اسم النشاط"""
    act = activity.strip()
    
    if any(x in act for x in ["تعبير", "قراءة", "تخطيط", "لغة"]):
        return "اللغوي"
    
    elif "رياضيات" in act:
        return "الرياضي"
    
    elif any(x in act for x in ["علمية", "تكنولوجيا"]):
        return "العلمي"
    
    elif any(x in act for x in ["إسلامية", "مدنية"]):
        return "الاجتماعي"
    
    elif any(x in act for x in ["مسرح", "رسم", "موسيقى", "إنشاد", "تشكيلية"]):
        return "الفني"
    
    elif any(x in act for x in ["بدنية", "إيقاعية", "رياضة"]):
        return "البدني والإيقاعي"
        
    return ""

# ---------------------------------------------------------
# 2. إعداد الصفحة
# ---------------------------------------------------------
st.set_page_config(page_title="المذكرة اليومية (مع المجالات)", layout="wide", page_icon="📝")
st.markdown("""<style>.main { direction: rtl; text-align: right; } h1, h2, h3, p, div { text-align: right; }</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. دوال المعالجة
# ---------------------------------------------------------
def extract_text_from_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip(): full_text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip().replace("\n", " ") for cell in row.cells if cell.text.strip()]
            if row_text: full_text.append(" | ".join(row_text))
    return "\n".join(full_text)

def analyze_with_cerebras(text, key, model_id):
    client = Cerebras(api_key=key)
    system_prompt = """
    أنت خبير تربوي. استخرج بيانات الدروس.
    المطلوب JSON List للكائنات:
    1. "النشاط": (رياضيات، تعبير شفوي، مبادئ القراءة، تخطيط، ت علمية، ت إسلامية، ت مدنية، ت بدنية، مسرح وعرائس، رسم وأشغال، ت إيقاعية، موسيقى وإنشاد).
    2. "الموضوع": عنوان الدرس.
    3. "الكفاءة": الكفاءة القاعدية.
    4. "المؤشر": مؤشر الكفاءة.
    """
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text[:25000]}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# 4. دالة إنشاء ملف Word (مع عمود المجال)
# ---------------------------------------------------------
def create_daily_journal(day_name, extracted_lessons):
    doc = Document()
    
    # إعداد الصفحة A4 Landscape ليكون الجدول عريضاً
    section = doc.sections[0]
    section.page_width = Inches(11.69)
    section.page_height = Inches(8.27)
    section.orientation = 1  # Landscape
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    
    # العنوان
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f'المذكرة اليومية - يوم: {day_name}')
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)

    # إنشاء الجدول (أضفنا عمود "المجال")
    headers = ["التوقيت", "النشاط", "المجال", "الموضوع (المحتوى)", "الكفاءة", "المؤشر", "ملاحظات"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.direction = WD_TABLE_DIRECTION.RTL
    table.autofit = False 
    
    # تنسيق الرأس
    hdr_cells = table.rows[0].cells
    widths = [0.8, 1.0, 0.9, 1.5, 1.2, 1.2, 0.8] # عرض الأعمدة بالبوصة
    
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        hdr_cells[i].width = Inches(widths[i])
        paragraph = hdr_cells[i].paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.runs[0]
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(255, 255, 255)
        # تلوين خلفية الرأس (محاكاة) - يتطلب مكتبات معقدة لذا نكتفي باللون
        
    # تجهيز البيانات
    day_schedule = WEEKLY_SCHEDULE.get(day_name, [])
    
    lessons_list = []
    if isinstance(extracted_lessons, dict):
        for val in extracted_lessons.values():
            if isinstance(val, list): lessons_list = val; break
        if not lessons_list: lessons_list = [extracted_lessons]
    else: lessons_list = extracted_lessons

    # تعبئة الجدول
    for slot in day_schedule:
        row_cells = table.add_row().cells
        
        # التوقيت
        row_cells[0].text = slot['time']
        
        # النشاط والمجال
        activity_name = slot['activity']
        row_cells[1].text = activity_name
        
        # حساب المجال تلقائياً
        domain_name = get_domain(activity_name)
        row_cells[2].text = domain_name
        
        # البحث عن الدرس
        found_lesson = None
        clean_slot = activity_name.replace("ت ", "").replace("مبادئ ", "").strip()
        
        for lesson in lessons_list:
            lesson_act = str(lesson.get('النشاط', '')).replace("ت ", "").replace("مبادئ ", "").strip()
            if clean_slot in lesson_act or lesson_act in clean_slot:
                found_lesson = lesson
                break
        
        if found_lesson:
            row_cells[3].text = str(found_lesson.get('الموضوع', ''))
            row_cells[4].text = str(found_lesson.get('الكفاءة', ''))
            row_cells[5].text = str(found_lesson.get('المؤشر', ''))
        else:
            row_cells[3].text = ""

        # تنسيق الخلايا
        for i, cell in enumerate(row_cells):
            cell.width = Inches(widths[i])
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if p.runs:
                    p.runs[0].font.size = Pt(10)
                    p.runs[0].font.name = "Arial"

    return doc

# ---------------------------------------------------------
# 5. الواجهة الرئيسية
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    default_key = st.secrets.get("CEREBRAS_API_KEY", "")
    api_key = st.text_input("Cerebras API Key", value=default_key, type="password")
    model_choice = st.selectbox("النموذج", ["llama-3.3-70b", "llama3.1-8b"])

st.title("📝 مولد المذكرة اليومية (مع المجالات)")

uploaded_file = st.file_uploader("📂 ملف المذكرات (.docx)", type=["docx"])
selected_day = st.selectbox("📅 اختر اليوم:", list(WEEKLY_SCHEDULE.keys()))

if uploaded_file and st.button("🚀 إنشاء"):
    if not api_key:
        st.error("أدخل المفتاح.")
    else:
        with st.spinner('جاري العمل...'):
            try:
                text = extract_text_from_docx(uploaded_file)
                data = analyze_with_cerebras(text, api_key, model_choice)
                if "error" not in data:
                    doc = create_daily_journal(selected_day, data)
                    bio = io.BytesIO()
                    doc.save(bio)
                    st.success("تم!")
                    st.download_button("📥 تحميل Word", bio.getvalue(), f"Journal_{selected_day}.docx")
                    with st.expander("البيانات"): st.json(data)
                else:
                    st.error(data["error"])
            except Exception as e:
                st.error(str(e))
