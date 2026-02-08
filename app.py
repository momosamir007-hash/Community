import streamlit as st
from cerebras.cloud.sdk import Cerebras

# 1. عنوان للصفحة
st.title("فحص موديلات Cerebras")

# 2. إدخال المفتاح
api_key = st.text_input("ضع المفتاح هنا:", type="password")

if st.button("ابحث عن الموديلات"):
    if not api_key:
        st.warning("الرجاء إدخال المفتاح أولاً.")
    else:
        try:
            client = Cerebras(api_key=api_key)
            
            st.info("جاري الاتصال بالسيرفر...")
            
            # جلب القائمة
            models_response = client.models.list()
            
            st.success("تم الاتصال! إليك الموديلات المتاحة:")
            
            # عرض كل موديل في بطاقة
            for m in models_response:
                st.code(m.id)  # استخدمنا st.code ليسهل عليك نسخه
                
        except Exception as e:
            st.error(f"حدث خطأ: {e}")
