import streamlit as st
from docx import Document
from transformers import pipeline
import io

# ---------------------------------------------------------
# 1. إعداد الصفحة وتصميمها
# ---------------------------------------------------------
st.set_page_config(
    page_title="المتلخص الذكي للمستندات",
    page_icon="📑",
    layout="centered",
    initial_sidebar_state="expanded"
)

# تخصيص CSS لدعم العربية (RTL) وتجميل الواجهة
st.markdown("""
<style>
    .main { text-align: right; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; text-align: right; }
    .stMarkdown, p, div { text-align: right; direction: rtl; }
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; }
    .stDownloadButton>button { width: 100%; background-color: #008CBA; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📑 تلخيص ملفات Word بالذكاء الاصطناعي")
st.markdown("قم برفع ملف `.docx` وسيقوم النموذج باستخراج العناوين وتلخيص محتواها.")

# ---------------------------------------------------------
# 2. تحميل نموذج الذكاء الاصطناعي (Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """
    تحميل النموذج مرة واحدة فقط وحفظه في الذاكرة
    لتجنب إعادة التحميل مع كل ضغطة زر.
    """
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    summarizer = pipeline("summarization", model=model_name, device=-1) # device=-1 for CPU
    return summarizer

# تحميل النموذج مع مؤشر انتظار
with st.spinner('جاري تحميل نموذج الذكاء الاصطناعي... (يحدث مرة واحدة فقط)'):
    try:
        summarizer = load_model()
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
        st.stop()

# ---------------------------------------------------------
# 3. دوال المعالجة والتلخيص
# ---------------------------------------------------------
def summarize_text(text):
    clean_text = text.strip()
    if not clean_text:
        return "لا يوجد محتوى."
    
    words = clean_text.split()
    if len(words) < 30:
        return clean_text  # النص قصير جداً لا يحتاج تلخيص

    try:
        summary = summarizer(
            clean_text,
            max_length=100,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        return summary[0]['summary_text']
    except Exception:
        return "النص طويل جداً أو معقد، تم عرض جزء منه."

def process_docx(file):
    doc = Document(file)
    results = []
    current_title = "مقدمة / بدون عنوان"
    buffer = ""

    # شريط التقدم
    progress_bar = st.progress(0)
    total_paragraphs = len(doc.paragraphs)
    
    for i, para in enumerate(doc.paragraphs):
        # تحديث شريط التقدم
        if i % 10 == 0:
            progress_bar.progress(min(i / total_paragraphs, 1.0))

        if para.style.name.startswith("Heading"):
            # تلخيص ما سبق قبل الانتقال للعنوان الجديد
            if buffer.strip():
                summary = summarize_text(buffer)
                results.append({"title": current_title, "summary": summary})
            
            current_title = para.text
            buffer = ""
        else:
            buffer += para.text + " "

    # إضافة آخر قسم
    if buffer.strip():
        summary = summarize_text(buffer)
        results.append({"title": current_title, "summary": summary})
    
    progress_bar.progress(1.0)
    return results

def create_download_file(results):
    """إنشاء ملف نصي للنتائج للتحميل"""
    output_text = "ملخص المستند - تم بواسطة الذكاء الاصطناعي\n"
    output_text += "="*40 + "\n\n"
    
    for item in results:
        output_text += f"📌 العنوان: {item['title']}\n"
        output_text += f"📄 الملخص: {item['summary']}\n"
        output_text += "-"*40 + "\n"
    
    return output_text

# ---------------------------------------------------------
# 4. واجهة المستخدم الرئيسية
# ---------------------------------------------------------
uploaded_file = st.file_uploader("اختر ملف Word", type=["docx"])

if uploaded_file is not None:
    st.info(f"تم رفع الملف: {uploaded_file.name}")

    if st.button("🚀 ابدأ التحليل والتلخيص"):
        with st.spinner('جاري قراءة الملف وتلخيص الفقرات...'):
            try:
                results = process_docx(uploaded_file)
                
                st.success("تم الانتهاء من التلخيص!")
                st.divider()

                # عرض النتائج
                for item in results:
                    with st.expander(f"📌 {item['title']}", expanded=True):
                        st.write(item['summary'])
                
                # زر التحميل (Download)
                st.divider()
                download_str = create_download_file(results)
                st.download_button(
                    label="📥 تحميل الملخص كملف نصي (TXT)",
                    data=download_str,
                    file_name="summary_report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"حدث خطأ غير متوقع: {e}")
