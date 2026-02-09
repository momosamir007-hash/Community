import streamlit as st
from docx import Document
from transformers import pipeline
import io

# ---------------------------------------------------------
# 1. إعداد الصفحة وتصميمها (RTL للعربية)
# ---------------------------------------------------------
st.set_page_config(
    page_title="المتلخص الذكي للمستندات",
    page_icon="📑",
    layout="centered"
)

# تخصيص CSS لدعم العربية (RTL) بشكل كامل
st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    .stMarkdown, .stButton, .stDownloadButton, .stFileUploader, h1, h2, h3, p, div { 
        text-align: right; 
        direction: rtl; 
    }
    /* جعل النصوص داخل الصناديق محاذاة لليمين */
    .stAlert { direction: rtl; text-align: right; }
    .stExpander { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.title("📑 تلخيص ملفات Word بالذكاء الاصطناعي")
st.write("---")

# ---------------------------------------------------------
# 2. تحميل نموذج الذكاء الاصطناعي (Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """
    تحميل النموذج مرة واحدة فقط.
    تم تغيير المهمة إلى 'text2text-generation' لحل مشكلة Unknown task.
    """
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    # التصحيح الأساسي هنا: استخدام text2text-generation
    pipe = pipeline("text2text-generation", model=model_name)
    return pipe

# تحميل النموذج في الخلفية
try:
    with st.spinner('جاري تهيئة نموذج الذكاء الاصطناعي... (يرجى الانتظار دقيقة في المرة الأولى)'):
        summarizer = load_model()
except Exception as e:
    st.error(f"حدث خطأ في تحميل النموذج: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. دوال المعالجة
# ---------------------------------------------------------
def summarize_text(text):
    """دالة التلخيص مع معالجة الأخطاء"""
    clean_text = text.strip()
    if not clean_text:
        return "لا يوجد محتوى."
    
    words = clean_text.split()
    if len(words) < 30:
        return clean_text  # النص قصير جداً لا يحتاج تلخيص

    try:
        # mT5 يتطلب text2text-generation
        result = summarizer(
            clean_text,
            max_length=150,  # أقصى طول للملخص
            min_length=30,   # أقل طول للملخص
            do_sample=False,
            truncation=True  # قص النص إذا كان طويلاً جداً
        )
        # التصحيح الثاني: المفتاح هو generated_text
        return result[0]['generated_text']
    except Exception as e:
        return f"تعذر التلخيص: {e}"

def process_docx(file):
    """قراءة ملف Word وتقسيمه حسب العناوين"""
    doc = Document(file)
    results = []
    
    current_title = "مقدمة / بدون عنوان"
    buffer = ""

    # شريط التقدم
    progress_bar = st.progress(0)
    total_paragraphs = len(doc.paragraphs)
    if total_paragraphs == 0:
        total_paragraphs = 1
    
    for i, para in enumerate(doc.paragraphs):
        # تحديث شريط التقدم كل 10 فقرات
        if i % 10 == 0:
            progress_bar.progress(min(i / total_paragraphs, 1.0))

        if para.style.name.startswith("Heading"):
            # إذا وجدنا عنواناً جديداً، نلخص ما قبله
            if buffer.strip():
                summary = summarize_text(buffer)
                results.append({"title": current_title, "summary": summary})
            
            current_title = para.text
            buffer = ""
        else:
            buffer += para.text + " "

    # إضافة القسم الأخير المتبقي في الذاكرة
    if buffer.strip():
        summary = summarize_text(buffer)
        results.append({"title": current_title, "summary": summary})
    
    progress_bar.progress(1.0)
    return results

def create_download_file(results):
    """تجهيز ملف نصي للتحميل"""
    output = io.StringIO()
    output.write("تقرير التلخيص الآلي\n")
    output.write("===================\n\n")
    for item in results:
        output.write(f"📌 {item['title']}\n")
        output.write(f"{item['summary']}\n")
        output.write("-" * 30 + "\n")
    return output.getvalue()

# ---------------------------------------------------------
# 4. واجهة الرفع والعرض (الرئيسية)
# ---------------------------------------------------------

# زر الرفع (موجود دائماً في الواجهة)
uploaded_file = st.file_uploader("📂 اختر ملف Word (.docx)", type=["docx"])

if uploaded_file is not None:
    st.success(f"تم استلام الملف: {uploaded_file.name}")

    # زر البدء
    if st.button("🚀 ابدأ التحليل والتلخيص"):
        with st.spinner('جاري قراءة الملف وتلخيص الفقرات...'):
            try:
                # عملية المعالجة
                final_results = process_docx(uploaded_file)
                
                st.balloons() # احتفال بانتهاء العملية
                st.success("تم الانتهاء بنجاح!")
                st.write("---")

                # عرض النتائج
                for item in final_results:
                    with st.expander(f"📌 {item['title']}", expanded=True):
                        st.write(item['summary'])
                
                # تحميل النتائج
                st.write("---")
                txt_data = create_download_file(final_results)
                st.download_button(
                    label="📥 تحميل التقرير (TXT)",
                    data=txt_data,
                    file_name="summary_report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"حدث خطأ غير متوقع أثناء المعالجة: {e}")

# تذييل الصفحة
st.markdown("<br><br><p style='text-align:center; color:grey;'>تم التطوير باستخدام Streamlit & Transformers</p>", unsafe_allow_html=True)
