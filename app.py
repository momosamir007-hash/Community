import streamlit as st
import requests
import json

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="Cerebras 6 Models",
    page_icon="⚡",
    layout="centered"
)

# تخصيص CSS لدعم العربية
st.markdown("""
<style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stTextInput > div > div > input { direction: rtl; text-align: right; }
    p { text-align: right; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Cerebras: القوة السداسية")

# --- 2. إدارة المفتاح (Secrets) ---
try:
    api_key = st.secrets["CEREBRAS_API_KEY"]
    st.sidebar.success("✅ المفتاح متصل (Secrets)")
except (FileNotFoundError, KeyError):
    api_key = st.sidebar.text_input("أدخل مفتاح API:", type="password")

if not api_key:
    st.warning("الرجاء توفير مفتاح API للبدء.")
    st.stop()

# --- 3. القائمة الجانبية (الموديلات الـ 6) ---
with st.sidebar:
    st.header("🎛️ لوحة التحكم")
    
    # القائمة التي تحتوي على الموديلات الستة التي ظهرت لك
    models_list = [
        "llama-3.3-70b",        # الأقوى والأحدث
        "llama3.1-8b",          # السريع والخفيف
        "qwen-3-32b",           # ممتاز في البرمجة
        "gpt-oss-120b",         # موديل ضخم
        "zai-glm-4.7",          # موديل متخصص
        "qwen-3-235b-a22b-instruct-2507" # الموديل العملاق
    ]
    
    selected_model = st.selectbox("اختر الموديل:", models_list, index=0)
    
    st.info(f"الموديل الحالي: **{selected_model}**")
    
    system_prompt = st.text_area(
        "تعليمات النظام:",
        value="أنت مساعد ذكي ومفيد.",
        height=100
    )
    
    if st.button("🗑️ مسح الذاكرة"):
        st.session_state.messages = []
        st.rerun()

# --- 4. دالة الاتصال (Streaming) ---
def stream_chat(messages, api_key, model, system_prompt):
    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": True,
        "max_tokens": 2000,
        "temperature": 0.7
    }
    
    try:
        with requests.post(url, headers=headers, json=data, stream=True) as r:
            if r.status_code != 200:
                yield f"⚠️ خطأ: {r.text}"
                return
                
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8').replace("data: ", "")
                    if decoded.strip() == "[DONE]": break
                    try:
                        chunk = json.loads(decoded)
                        content = chunk['choices'][0]['delta'].get('content', '')
                        if content: yield content
                    except: continue
    except Exception as e:
        yield f"❌ خطأ: {e}"

# --- 5. واجهة الدردشة ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# الإدخال
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in stream_chat(st.session_state.messages, api_key, selected_model, system_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
