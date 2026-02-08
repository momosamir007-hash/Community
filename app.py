import streamlit as st
import requests
import json

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="Cerebras AI",
    page_icon="⚡",
    layout="centered"
)

# تخصيص CSS لدعم اللغة العربية وتنسيق المحادثة
st.markdown("""
<style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stTextInput > div > div > input { direction: rtl; text-align: right; }
    .stTextArea > div > div > textarea { direction: rtl; text-align: right; }
    p { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Cerebras: الذكاء الخارق")

# --- 2. جلب المفتاح من Secrets (الخطوة الذكية) ---
try:
    # يحاول قراءة المفتاح من أسرار Streamlit
    api_key = st.secrets["CEREBRAS_API_KEY"]
    st.sidebar.success("✅ المفتاح متصل بأمان (Secrets)")
except (FileNotFoundError, KeyError):
    # في حال كنت تجرب محلياً ولم تضبط الأسرار، يطلب المفتاح يدوياً
    st.sidebar.warning("⚠️ لم يتم العثور على المفتاح في Secrets")
    api_key = st.sidebar.text_input("أدخل المفتاح يدوياً للتجربة:", type="password")

# إذا لم يتوفر المفتاح بأي طريقة، نوقف التطبيق
if not api_key:
    st.info("الرجاء إعداد CEREBRAS_API_KEY في إعدادات Streamlit Cloud.")
    st.stop()

# --- 3. القائمة الجانبية للإعدادات ---
with st.sidebar:
    st.markdown("---")
    
    # اختيار الموديل
    model = st.selectbox(
        "🧠 اختر الموديل:",
        ["llama-3.3-70b", "llama3.1-8b", "qwen-3-32b"],
        index=0
    )
    
    # شخصية البوت
    system_prompt = st.text_area(
        "🎭 دور المساعد:",
        value="أنت مساعد ذكي ومفيد، تتحدث اللغة العربية بطلاقة ووضوح.",
        height=100
    )
    
    # زر مسح الذاكرة
    if st.button("🗑️ محادثة جديدة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 4. دالة الاتصال (Streaming Engine) ---
def stream_cerebras_api(messages, api_key, model, system_prompt):
    url = "https://api.cerebras.ai/v1/chat/completions"
    
    # دمج تعليمات النظام
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": full_messages,
        "temperature": 0.7,
        "max_tokens": 1500,
        "stream": True 
    }
    
    try:
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            if response.status_code != 200:
                yield f"⚠️ خطأ: {response.text}"
                return

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:] 
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except:
                            continue
    except Exception as e:
        yield f"❌ خطأ في الاتصال: {e}"

# --- 5. منطق المحادثة ---

# تهيئة الذاكرة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل القديمة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# استقبال السؤال الجديد
if prompt := st.chat_input("سألني أي شيء..."):
    
    # عرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # عرض الجواب
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # استدعاء الدالة
        for chunk in stream_cerebras_api(st.session_state.messages, api_key, model, system_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    # حفظ الجواب
    st.session_state.messages.append({"role": "assistant", "content": full_response})
