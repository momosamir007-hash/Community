

import streamlit as st

# عنوان التطبيق
st.title("تطبيق Streamlit الأول لي 🚀")

# إدخال النص
name = st.text_input("ما هو اسمك؟")

# زر لعرض الرسالة
if st.button("قل مرحباً"):
    if name:
        st.success(f"مرحباً {name}! أهلاً بك في عالم Streamlit.")
    else:
        st.warning("الرجاء إدخال اسمك أولاً.")

# قسم للحسابات
st.header("آلة حاسبة بسيطة")
num1 = st.number_input("الرقم الأول", value=0)
num2 = st.number_input("الرقم الثاني", value=0)
operation = st.selectbox("اختر العملية", ["جمع", "طرح", "ضرب", "قسمة"])

if st.button("احسب"):
    if operation == "جمع":
        result = num1 + num2
    elif operation == "طرح":
        result = num1 - num2
    elif operation == "ضرب":
        result = num1 * num2
    else:
        if num2 != 0:
            result = num1 / num2
        else:
            result = "لا يمكن القسمة على صفر"
    st.write(f"النتيجة: {result}")

# عرض معلومات إضافية
st.sidebar.title("عن التطبيق")
st.sidebar.info("تم إنشاء هذا التطبيق باستخدام Streamlit ونشره على Streamlit Cloud.")
