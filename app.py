import streamlit as st
from docx import Document
from cerebras.cloud.sdk import Cerebras
import pandas as pd
import json
import io

# ---------------------------------------------------------
# 1. إعداد الصفحة وتصميمها
# ---------------------------------------------------------
st.set_page_config(
    page_title="مستخرج البيانات التربوية (Cerebras)",
    page_icon="🚀",
    layout="wide"
)

# تخصيص CSS للعربية
st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    .stMarkdown, .stButton, .stDownloadButton, .stFileUploader, h1, h2, h3, p, div, label, input { 
        text-align: right; 
        direction: rtl; 
    }
    .stDataFrame { direction: ltr; } 
    [data-testid="stSidebar"] { text-align: right; direction: rtl; }
    
    /* تنسيق خاص لرسائل الخطأ والنجاح */
    .stSuccess, .stError, .stWarning { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. الشريط الجانبي: إعدادات API
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    api_key = st.text_input("Cerebras API Key", type="password", help="أدخل مفتاح Cerebras الخاص بك هنا")
    
    # اختيار النموذج (Cerebras يدعم Llama بشكل ممتاز)
    model_choice = st.selectbox(
        "اختر النموذج",
        ["llama3.1-70b", "llama-3.3-70b"],
        index=0
    )
    
    st.info("يتميز Cerebras بسرعة فائقة في معالجة النصوص الطويلة.")

# ---------------------------------------------------------
# 3. دوال المعالجة
# ---------------------------------------------------------
def extract_text_from_docx(file):
    """قراءة كل النصوص داخل الملف (فقرات + جداول) لضمان عدم ضياع أي معلومة"""
    doc = Document(file)
    full_text = []
    
    # 1. قراءة الفقرات العادية
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
            
    # 2. قراءة الجداول (الأهم في المذكرات التربوية)
    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                # تنظيف النص داخل الخلية
                cell_text = cell.text.strip().replace("\n", " ")
                if cell_text:
                    row_text.append(cell_text)
            if row_text:
                # دمج خلايا الصف بفاصل مميز
                full_text.append(" | ".join(row_text))
                
    return "\n".join(full_text)

def analyze_with_cerebras(text, key, model_id):
    """إرسال النص لنموذج Cerebras لاستخراج البيانات JSON"""
    
    client = Cerebras(api_key=key)
    
    # هندسة الأوامر (Prompt Engineering) دقيقة جداً
    system_prompt = """
    أنت مساعد إداري تربوي خبير في تحليل المذكرات التربوية الجزائرية.
    مهمتك هي استخراج بيانات الأنشطة التربوية من النص المقدم بدقة عالية.
    
    يجب أن تستخرج البيانات التالية لكل نشاط تجده:
    1. "النشاط": (مثال: تعبير شفوي، رياضيات، تربية إسلامية...)
    2. "الموضوع": (عنوان الدرس)
    3. "الكفاءة_القاعدية": (نص الكفاءة المستهدفة)
    4. "مؤشر_الكفاءة": (مؤشر واحد أو أكثر)

    القواعد الصارمة:
    - المخرج يجب أن يكون JSON Valid فقط.
    - التنسيق: قائمة من الكائنات (List of Objects).
    - لا تضف أي نص قبل أو بعد الـ JSON (مثل "Here is the code").
    - إذا كانت المعلومة مفقودة، اكتب "غير مذكور".
    - النص يحتوي على جداول تم تحويلها لنص، حاول فهم السياق.
    """

    user_prompt = f"""
    استخرج البيانات من النص التالي:
    
    {text[:25000]} 
    """ 
    # Cerebras يدعم سياق كبير، لكن نحدد 25000 حرف للأمان

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # حرارة منخفضة للدقة
            max_tokens=4000,
            response_format={"type": "json_object"} # إجبار النموذج على إخراج JSON
        )
        
        response_content = completion.choices[0].message.content
        
        # التأكد من أن النص هو JSON صافي
        return json.loads(response_content)
        
    except json.JSONDecodeError:
        return {"error": "فشل النموذج في إرجاع تنسيق JSON صحيح. حاول مرة أخرى."}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# 4. واجهة المستخدم الرئيسية
# ---------------------------------------------------------
st.title("🚀 استخراج المذكرات التربوية (Cerebras AI)")
st.markdown("""
هذا التطبيق يستخدم **Cerebras** لاستخراج:
- **النشاط**
- **الموضوع**
- **الكفاءة القاعدية**
- **مؤشر الكفاءة**
""")

uploaded_file = st.file_uploader("📂 اختر ملف Word (.docx)", type=["docx"])

if uploaded_file and api_key:
    if st.button("⚡ ابدأ التحليل السريع"):
        with st.spinner('جاري قراءة الملف وتحليل البيانات بسرعة البرق...'):
            try:
                # 1. استخراج النص
                raw_text = extract_text_from_docx(uploaded_file)
                
                # 2. التحليل بالذكاء الاصطناعي
                # نتوقع أن يعود JSON يحتوي على مفتاح رئيسي مثل "lessons" أو قائمة مباشرة
                result = analyze_with_cerebras(raw_text, api_key, model_choice)
                
                # معالجة هيكل الـ JSON العائد (قد يكون قائمة مباشرة أو داخل مفتاح)
                data_list = []
                if isinstance(result, list):
                    data_list = result
                elif isinstance(result, dict):
                    # البحث عن القائمة داخل القاموس
                    for key, value in result.items():
                        if isinstance(value, list):
                            data_list = value
                            break
                    # إذا لم نجد قائمة، ربما القاموس نفسه هو عنصر واحد
                    if not data_list and "النشاط" in result:
                        data_list = [result]
                
                if data_list:
                    st.success(f"تم استخراج {len(data_list)} نشاط بنجاح!")
                    
                    # 3. تحويل إلى جدول وعرضه
                    df = pd.DataFrame(data_list)
                    
                    # إعادة ترتيب الأعمدة لتكون منطقية
                    cols_order = ["النشاط", "الموضوع", "الكفاءة_القاعدية", "مؤشر_الكفاءة"]
                    # نختار فقط الأعمدة الموجودة فعلياً
                    final_cols = [c for c in cols_order if c in df.columns]
                    # نضيف باقي الأعمدة إن وجدت
                    remaining_cols = [c for c in df.columns if c not in final_cols]
                    df = df[final_cols + remaining_cols]

                    st.dataframe(df, use_container_width=True)
                    
                    # 4. خيارات التحميل
                    col1, col2 = st.columns(2)
                    
                    # تحميل Excel
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='المذكرات')
                        
                    with col1:
                        st.download_button(
                            label="📥 تحميل كملف Excel",
                            data=buffer.getvalue(),
                            file_name="lesson_plans_cerebras.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    # تحميل JSON
                    json_str = json.dumps(data_list, ensure_ascii=False, indent=4)
                    with col2:
                        st.download_button(
                            label="📥 تحميل كملف JSON",
                            data=json_str,
                            file_name="lesson_plans.json",
                            mime="application/json"
                        )
                
                elif "error" in result:
                    st.error(f"خطأ: {result['error']}")
                else:
                    st.warning("لم يتم العثور على بيانات. تحقق من محتوى الملف.")
                    
            except Exception as e:
                st.error(f"حدث خطأ غير متوقع: {e}")

elif uploaded_file and not api_key:
    st.warning("⚠️ يرجى إدخال مفتاح Cerebras API في القائمة الجانبية.")

# تذييل
st.write("---")
st.markdown("<p style='text-align:center; color:grey;'>Powered by Cerebras Llama-3.1-70b</p>", unsafe_allow_html=True)
