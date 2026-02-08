import streamlit as st
from cerebras.cloud.sdk import Cerebras

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="مشروعي الذكي", page_icon="🚀")
st.title("🤖 مساعد ذكي سريع (Cerebras)")

# --- 2. إعداد الاتصال بـ Cerebras ---
# استبدل النص أدناه بمفتاحك الحقيقي
API_KEY = "ضع_مفتاح_CEREBRAS_هنا"

# التحقق من وجود المفتاح
if not API_KEY or API_KEY == "csk-j9hy4epdhjft3tntdvcmd99498xhd2v36w4ym8wn9vy6mhnm":
    st.error("الرجاء وضع الـ API Key في الكود لتشغيل التطبيق.")
    st.stop()

client = Cerebras(api_key=API_KEY)

# --- 3. ذاكرة المحادثة (Session State) ---
# هذه الخطوة مهمة لكي "يتذكر" البوت سياق الحديث السابق
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. عرض الرسائل القديمة في الشاشة ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. استقبال المدخلات والرد ---
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    
    # أ. عرض رسالة المستخدم فوراً
    with st.chat_message("user"):
        st.markdown(prompt)
    # حفظ رسالة المستخدم في الذاكرة
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ب. تجهيز الرد من الذكاء الاصطناعي
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # مكان فارغ للنص المتدفق
        full_response = ""
        
        try:
            # إرسال الطلب مع الذاكرة الكاملة (messages)
            stream = client.chat.completions.create(
                model="llama3.1-70b", # الموديل الذكي والسريع
                messages=[
                    {"role": "system", "content": "أنت مساعد مفيد وتتحدث العربية بوضوح."}
                ] + st.session_state.messages, # نرسل التاريخ السابق
                stream=True,
            )
            
            # استقبال الرد كلمة بكلمة (Streaming)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    message_placeholder.markdown(full_response + "▌") # تأثير المؤشر
            
            message_placeholder.markdown(full_response)
            
            # حفظ رد البوت في الذاكرة
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"حدث خطأ: {e}")
