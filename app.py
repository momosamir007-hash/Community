import streamlit as st
import requests
import json

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Cerebras Debugger", page_icon="🛠️")

# تخصيص CSS
st.markdown("""
<style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stTextInput > div > div > input { direction: rtl; text-align: right; }
    .stSelectbox > div > div > div { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ فحص موديلات Cerebras")

# --- 2. المفتاح ---
try:
    api_key = st.secrets["CEREBRAS_API_KEY"]
except:
    api_key = st.sidebar.text_input("API Key:", type="password")

if not api_key:
    st.warning("الرجاء إدخال المفتاح.")
    st.stop()

# --- 3. القائمة ---
with st.sidebar:
    st.header("اختيار الموديل")
    
    # القائمة الكاملة التي ظهرت لك
    models = [
        "llama-3.3-70b",   # ✅ (ممتاز ومستقر)
        "llama3.1-8b",     # ✅ (سريع جداً)
        "qwen-3-32b",      # ❓ (جرب)
        "gpt-oss-120b",    # ⚠️ (غالباً تجريبي)
        "zai-glm-4.7",     # ⚠️ (قد لا يعمل)
        "qwen-3-235b-a22b-instruct-2507" # ⚠️ (اسم معقد قد يتغير)
    ]
    
    selected_model = st.radio("اختر موديل للتجربة:", models)
    
    if st.button("🗑️ مسح الذاكرة"):
        st.session_state.messages = []
        st.rerun()

# --- 4. الدالة مع كشف الأخطاء التفصيلي ---
def stream_chat_debug(messages, api_key, model):
    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        
        # إذا كان هناك خطأ من السيرفر (ليس 200)
        if response.status_code != 200:
            error_details = response.text
            try:
                # محاولة قراءة الخطأ بصيغة JSON ليكون أوضح
                error_json = response.json()
                error_msg = error_json.get('error', {}).get('message', error_details)
                yield f"⛔ **فشل الموديل:** {model}\n\n**السبب:** {error_msg}"
            except:
                yield f"⛔ **خطأ غير معروف:** رمز الحالة {response.status_code}\n{error_details}"
            return

        # إذا نجح الاتصال، ابدأ البث
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8').replace("data: ", "")
                if decoded.strip() == "[DONE]": break
                try:
                    chunk = json.loads(decoded)
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content: yield content
                except: continue
                
    except Exception as e:
        yield f"❌ خطأ في الاتصال بالإنترنت: {e}"

# --- 5. التشغيل ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("جرب الموديل بكلمة 'مرحبا'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_holder = st.empty()
        full_text = ""
        
        # استدعاء دالة الفحص
        for chunk in stream_chat_debug(st.session_state.messages, api_key, selected_model):
            full_text += chunk
            response_holder.markdown(full_text + "▌")
        
        response_holder.markdown(full_text)
        
        # إذا كان الرد رسالة خطأ، لا نحفظه في الذاكرة لكي لا يفسد المحادثة التالية
        if "⛔" not in full_text:
            st.session_state.messages.append({"role": "assistant", "content": full_text})

