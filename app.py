import streamlit as st
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_DIRECTION
from cerebras.cloud.sdk import Cerebras
import json
import io

# ---------------------------------------------------------
# 1. إعداد التوقيت الأسبوعي (تم نقله حرفياً من الصورة 1)
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
# 2. إعداد الصفحة
# ---------------------------------------------------------
st.set_page_config(page_title="المذكرة اليومية الآلية", layout="wide", page_icon="📅")
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
    
    # تحسين البرومبت لضمان استخراج الأسماء كما هي في الجدول
    system_prompt = """
    أنت خبير تربوي. استخرج بيانات الدروس من النص المرفق لملئها في المذكرة اليومية.
    
    يجب أن تستخرج قائمة JSON تحتوي على الكائنات التالية لكل درس:
    1. "النشاط": (حاول أن تطابق الاسم مع أحد هذه: رياضيات، تعبير شفوي، مبادئ القراءة، تخطيط، ت علمية، ت إسلامية، ت مدنية، ت بدنية، مسرح وعرائس، رسم وأشغال، ت إيقاعية، موسيقى وإنشاد).
    2. "الموضوع": عنوان الدرس بدقة.
    3. "الكفاءة": الكفاءة القاعدية/المستهدفة.
    4. "المؤشر": مؤشر الكفاءة.

    ملاحظة هامة:
    - إذا وجدت نشاط "قراءة" اكتبه "مبادئ القراءة".
    - إذا وجدت "تربية علمية" اكتبها "ت علمية".
    
    المخرج يجب أن يكون JSON valid فقط.
    """
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text[:25000]}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# 4. دالة إنشاء ملف Word (التصميم النهائي)
# ---------------------------------------------------------
def create_daily_journal(day_name, extracted_lessons):
    doc = Document()
    
    # هوامش الصفحة
    section = doc.sections[0]
    section.page_width = Inches(8.27) 
    section.page_height = Inches(11.69)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    
    # العنوان
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f'المذكرة اليومية - يوم: {day_name}')
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102) # أزرق غامق

    # إنشاء الجدول (6 أعمدة كما في الصورة 2)
    headers = ["التوقيت", "النشاط", "الموضوع (المحتوى)", "الكفاءة", "المؤشر", "ملاحظات"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.direction = WD_TABLE_DIRECTION.RTL 
    table.autofit = False 
    
    # تنسيق رأس الجدول
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        paragraph = hdr_cells[i].paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.runs[0]
        run.font.bold = True
        run.font.size = Pt(11)
        
        # تعيين عرض الأعمدة تقريبياً
        if i == 0: hdr_cells[i].width = Inches(0.8) # التوقيت
        elif i == 1: hdr_cells[i].width = Inches(1.0) # النشاط
        elif i == 2: hdr_cells[i].width = Inches(1.5) # الموضوع
        else: hdr_cells[i].width = Inches(1.2)

    # تجهيز البيانات
    day_schedule = WEEKLY_SCHEDULE.get(day_name, [])
    
    # استخراج القائمة من رد الـ AI
    lessons_list = []
    if isinstance(extracted_lessons, dict):
        # البحث عن أي مفتاح يحتوي على قائمة
        for key, val in extracted_lessons.items():
            if isinstance(val, list):
                lessons_list = val
                break
        if not lessons_list: 
             # ربما الـ JSON هو القائمة مباشرة أو داخل مفتاح غير متوقع
             lessons_list = [extracted_lessons]
    elif isinstance(extracted_lessons, list):
        lessons_list = extracted_lessons

    # تعبئة الجدول
    for slot in day_schedule:
        row_cells = table.add_row().cells
        
        # 1. التوقيت والنشاط (ثابت من الجدول)
        row_cells[0].text = slot['time']
        row_cells[1].text = slot['activity']
        
        # 2. البحث الذكي (Fuzzy Matching)
        found_lesson = None
        slot_activity_clean = slot['activity'].replace("ت ", "").replace("مبادئ ", "").strip()
        
        for lesson in lessons_list:
            lesson_act = str(lesson.get('النشاط', '')).replace("ت ", "").replace("مبادئ ", "").strip()
            
            # مطابقة جزئية
            if slot_activity_clean in lesson_act or lesson_act in slot_activity_clean:
                found_lesson = lesson
                break
        
        # 3. ملء البيانات
        if found_lesson:
            row_cells[2].text = str(found_lesson.get('الموضوع', ''))
            row_cells[3].text = str(found_lesson.get('الكفاءة', ''))
            row_cells[4].text = str(found_lesson.get('المؤشر', ''))
        else:
            # ترك فراغ للكتابة اليدوية
            row_cells[2].text = ""

        # تنسيق النصوص داخل الجدول
        for cell in row_cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if paragraph.runs:
                    paragraph.runs[0].font.size = Pt(10)
                    paragraph.runs[0].font.name = "Arial"

    return doc

# ---------------------------------------------------------
# 5. الواجهة الرئيسية
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    # محاولة جلب المفتاح تلقائياً
    default_key = st.secrets.get("CEREBRAS_API_KEY", "")
    api_key = st.text_input("Cerebras API Key", value=default_key, type="password")
    
    # *** التحديث هنا: استخدام النماذج الصحيحة ***
    model_choice = st.selectbox(
        "اختر النموذج",
        ["llama-3.3-70b", "llama3.1-8b"], 
        index=0
    )

st.title("📝 مولد المذكرة اليومية (النسخة المصححة)")
st.info("تم تحديث أسماء النماذج لحل مشكلة الخطأ 404.")

uploaded_file = st.file_uploader("📂 اختر ملف المذكرات (.docx)", type=["docx"])
selected_day = st.selectbox("📅 اختر اليوم:", list(WEEKLY_SCHEDULE.keys()))

if uploaded_file and st.button("🚀 إنشاء المذكرة"):
    if not api_key:
        st.error("⛔ يرجى إدخال مفتاح API.")
    else:
        with st.spinner(f'جاري تحليل الدروس لـ يوم {selected_day}...'):
            try:
                # 1. القراءة
                text_content = extract_text_from_docx(uploaded_file)
                
                # 2. التحليل (AI)
                ai_data = analyze_with_cerebras(text_content, api_key, model_choice)
                
                if "error" in ai_data:
                    st.error(f"حدث خطأ في الاتصال: {ai_data['error']}")
                else:
                    # 3. التوليد (Word)
                    doc_output = create_daily_journal(selected_day, ai_data)
                    
                    # حفظ في الذاكرة
                    buffer = io.BytesIO()
                    doc_output.save(buffer)
                    buffer.seek(0)
                    
                    st.success("✅ تم الإنشاء بنجاح!")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.download_button(
                            label=f"📥 تحميل مذكرة {selected_day}",
                            data=buffer,
                            file_name=f"Journal_{selected_day}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    
                    with col2:
                        with st.expander("عرض البيانات التي تم استخراجها (Debug)"):
                            st.json(ai_data)
                            
            except Exception as e:
                st.error(f"خطأ غير متوقع: {e}")
