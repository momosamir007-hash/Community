import streamlit as st
from cerebras.cloud.sdk import Cerebras

# 1. إعدادات الصفحة (يجب أن تكون أول سطر)
st.set_page_config(page_title="Cerebras Chat", page_icon="⚡", layout="centered")

st.title("⚡ Cerebras Fast Chat")
st.caption("مدعوم بواسطة نموذج Llama-3.1-70b وسرعة Cerebras")

# 2. القائمة الجانبية (Sidebar) لإعدادات المفتاح
with st.sidebar:
    st.header("الإعدادات")
    api_key = st.text_input("أدخل مفتاح Cerebras API:", type="password")
    st.markdown("[احصل على مفتاح مجاني من هنا](https://cloud.cerebras.ai/)")
    
    # زر لمسح المحادثة
    if st.button("مسح المحادثة 🗑️"):
        st.session_state.messages = []
        st.rerun()

# 3. التحقق من وجود المفتاح قبل البدء
if not api_key:
    st.info("الرجاء إدخال API Key في القائمة الجانبية للمتابعة.")
    st.stop()

# إنشاء العميل باستخدام المفتاح المدخل
client = Cerebras(api_key=api_key)

# 4. إدارة ذاكرة المحادثة (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "مرحباً! كيف يمكنني مساعدتك بسرعة اليوم؟"}]

# 5. عرض الرسائل السابقة في الشاشة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. معالجة المدخلات الجديدة
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    # أ. عرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ب. إرسال الطلب واستقبال الرد (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model="llama3.1-70b",
                messages=[
                    {"role": "system", "content": "أنت مساعد ذكي ومفيد."}
                ] + [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            # بناء الرد كلمة بكلمة
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"حدث خطأ: {e}")
            full_response = "عذراً، حدث خطأ في الاتصال."

    # ج. حفظ رد المساعد في الذاكرة
    st.session_state.messages.append({"role": "assistant", "content": full_response})
