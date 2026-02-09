import streamlit as st
from docx import Document
from cerebras.cloud.sdk import Cerebras
import pandas as pd
import json
import io
import time

# ---------------------------------------------------------
# 1. إعداد الصفحة وتصميمها (CSS محسن للغة العربية)
# ---------------------------------------------------------
st.set_page_config(
    page_title="المحلل التربوي الذكي",
    page_icon="🎓",
    layout="wide"
)

# تصميم CSS مخصص لجعل الواجهة عصرية ودعم العربية بالكامل
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    /* تنسيق العناوين */
    h1, h2, h3 {
        color: #2E86C1;
        font-weight: 700;
    }
    
    /* تنسيق الزر */
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
    }

    /* تنسيق الجدول */
    [data-testid="stDataFrame"] {
        direction: rtl;
        text-align: right;
    }
    
    /* رسائل التنبيه */
    .stSuccess, .stError, .stWarning {
        direction: rtl;
        border-radius: 10px;
    }
    
    /* القائمة الجانبية */
    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-left: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. الدوال المساعدة
# ---------------------------------------------------------

def extract_text_from_docx(file):
    """استخراج النصوص بذكاء مع الحفاظ على الهيكل العام"""
    doc = Document(file)
    full_text = []
    
    # استخراج الفقرات
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
            
    # استخراج الجداول (مفيد جداً في المذكرات)
    for table in doc.tables:
        for row in table.rows:
            row_data = [cell.text.strip().replace("\n", " ") for cell in row.cells if cell.text.strip()]
            if row_data:
                full_text.append(" | ".join(row_data))
                
    return "\n".join(full_text)

def analyze_with_cerebras(text, key, model_id):
    """تحليل النص واستخراج البيانات الهيكلية"""
    client = Cerebras(api_key=key)
    
    system_prompt = """
    أنت خبير تربوي ومحلل بيانات. مهمتك هي استخراج هيكلة الدروس من ملفات المذكرات التربوية.
    
    قم بتحليل النص واستخرج قائمة (JSON List) تحتوي على الكائنات التالية لكل نشاط تعليمي:
    1. "المجال_أو_المقطع": (العنوان الكبير، مثل: المجال اللغوي، الحياة المدرسية، أو اسم المقطع).
    2. "النشاط": (نوع الحصة، مثل: قراءة، رياضيات، تربية إسلامية).
    3. "الموضوع": (عنوان الدرس الدقيق).
    4. "الكفاءة_الختامية": (أو الكفاءة القاعدية).
    5. "المؤشر": (مؤشر الكفاءة أو الهدف التعلمي).
    
    ملاحظات هامة:
    - المخرج يجب أن يكون JSON Valid فقط بدون أي نصوص إضافية.
    - إذا كانت المعلومة غير موجودة صراحة، حاول استنتاجها من السياق أو اكتب "غير محدد".
    - رتب البيانات بدقة.
    """
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"النص المراد تحليله:\n{text[:28000]}"} # زيادة الحد المسموح قليلاً
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# 3. الواجهة الرئيسية
# ---------------------------------------------------------

# --- الشريط الجانبي ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063032.png", width=80)
    st.title("الإعدادات")
    
    # إدارة المفتاح
    if "CEREBRAS_API_KEY" in st.secrets:
        api_key = st.secrets["CEREBRAS_API_KEY"]
        st.success("🔑 المفتاح نشط (Secrets)")
    else:
        api_key = st.text_input("أدخل مفتاح API (Cerebras)", type="password")
    
    st.markdown("---")
    model_choice = st.selectbox("🧠 نموذج الذكاء الاصطناعي", ["llama-3.3-70b", "llama3.1-8b"])
    st.caption("يُنصح باستخدام Llama 3.3 للدقة العالية.")

# --- المحتوى الرئيسي ---
st.title("🎓 المستخرج الآلي للمذكرات التربوية")
st.markdown("##### ⚡ تحويل ملفات Word إلى جداول منظمة (Excel/JSON) بدقة عالية.")

# رفع الملف
uploaded_file = st.file_uploader("قم بسحب وإفلات ملف المذكرات (DOCX) هنا", type=["docx"])

if uploaded_file:
    # حاوية لعرض حالة الملف
    file_container = st.container()
    
    if st.button("🚀 بدء التحليل والاستخراج"):
        if not api_key:
            st.error("⚠️ يرجى إدخال مفتاح API في القائمة الجانبية.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # خطوة 1: قراءة الملف
                status_text.text("📂 جاري قراءة الملف...")
                progress_bar.progress(25)
                raw_text = extract_text_from_docx(uploaded_file)
                
                # خطوة 2: المعالجة
                status_text.text("🤖 الذكاء الاصطناعي يقوم بتحليل البيانات...")
                progress_bar.progress(60)
                result = analyze_with_cerebras(raw_text, api_key, model_choice)
                
                progress_bar.progress(90)
                
                # خطوة 3: عرض النتائج
                final_data = []
                if isinstance(result, dict):
                    # البحث عن القائمة داخل الـ JSON
                    for key, val in result.items():
                        if isinstance(val, list):
                            final_data = val
                            break
                    if not final_data and "error" not in result:
                         # ربما الرد هو كائن واحد فقط
                         final_data = [result]
                elif isinstance(result, list):
                    final_data = result
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                if "error" in result:
                    st.error(f"❌ حدث خطأ في المعالجة: {result['error']}")
                elif not final_data:
                    st.warning("⚠️ لم يتم العثور على بيانات مهيكلة. تأكد من محتوى الملف.")
                else:
                    # --- عرض النتائج بنجاح ---
                    st.success(f"✅ تم استخراج {len(final_data)} عنصراً بنجاح!")
                    
                    # إنشاء DataFrame
                    df = pd.DataFrame(final_data)
                    
                    # ترتيب الأعمدة (جعل العنوان والمجال في البداية)
                    cols_order = ["المجال_أو_المقطع", "النشاط", "الموضوع", "الكفاءة_الختامية", "المؤشر"]
                    # التأكد من وجود الأعمدة
                    existing_cols = [c for c in cols_order if c in df.columns]
                    remaining_cols = [c for c in df.columns if c not in existing_cols]
                    df = df[existing_cols + remaining_cols]

                    # عرض تفاعلي (Data Editor) يسمح بالتعديل
                    st.markdown("### 📝 مراجعة البيانات (يمكنك التعديل مباشرة في الجدول)")
                    edited_df = st.data_editor(
                        df,
                        use_container_width=True,
                        num_rows="dynamic",
                        column_config={
                            "المجال_أو_المقطع": st.column_config.TextColumn("المجال / الوحدة", help="العنوان الرئيسي أو الميدان"),
                            "النشاط": st.column_config.TextColumn("النشاط", width="small"),
                            "الموضوع": st.column_config.TextColumn("عنوان الدرس", width="medium"),
                            "الكفاءة_الختامية": st.column_config.TextColumn("الكفاءة", width="large"),
                        }
                    )
                    
                    st.markdown("---")
                    
                    # --- منطقة التحميل ---
                    st.subheader("📥 تحميل البيانات")
                    c1, c2, c3 = st.columns(3)
                    
                    # تحميل Excel (مع تنسيق بسيط)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        edited_df.to_excel(writer, index=False, sheet_name='Educational_Data')
                        # تجميل تلقائي لأعمدة اكسل
                        worksheet = writer.sheets['Educational_Data']
                        for column_cells in worksheet.columns:
                            length = max(len(str(cell.value)) for cell in column_cells)
                            worksheet.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 50)
                            
                    c1.download_button(
                        label="تحميل ملف Excel 📗",
                        data=buffer.getvalue(),
                        file_name="extracted_lessons.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # تحميل CSV
                    c2.download_button(
                        label="تحميل ملف CSV 📄",
                        data=edited_df.to_csv(index=False).encode('utf-8-sig'),
                        file_name="extracted_lessons.csv",
                        mime="text/csv"
                    )
                    
                    # تحميل JSON
                    c3.download_button(
                        label="تحميل ملف JSON ⚙️",
                        data=json.dumps(final_data, ensure_ascii=False, indent=4),
                        file_name="extracted_lessons.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"حدث خطأ غير متوقع: {e}")

else:
    # عرض رسالة ترحيبية عند عدم وجود ملف
    st.info("👆 ابدأ برفع ملف المذكرات من الأعلى.")
