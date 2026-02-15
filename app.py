import streamlit as st
import requests
import json

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Multi-Model Debugger", page_icon="🛠️")

# تخصيص CSS
st.markdown("""
<style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stTextInput > div > div > input { direction: rtl; text-align: right; }
    .stSelectbox > div > div > div { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ فحص موديلات (Cerebras + GLM)")

# --- 2. المفاتيح (Keys) ---
# محاولة جلب مفتاح Cerebras
try:
    cerebras_key = st.secrets["CEREBRAS_API_KEY"]
except:
    cerebras_key = st.sidebar.text_input("Cerebras API Key:", type="password")

# محاولة جلب مفتاح GLM
try:
    glm_key = st.secrets["GLM_API_KEY"]
except:
    glm_key = st.sidebar.text_input("GLM (Zhipu) API Key:", type="password")

# التحقق من وجود المفاتيح قبل المتابعة (تحذير فقط)
if not cerebras_key and not glm_key:
    st.warning("الرجاء إدخال مفتاح API واحد على الأقل.")
    st.stop()

# --- 3. القائمة ---
with st.sidebar:
    st.header("اختيار الموديل")
    
    # القائمة المحدثة
    models = [
        "llama-3.3-70b",   # Cerebras
        "llama3.1-8b",     # Cerebras
        "glm-4",           # GLM (ZhipuAI) - الموديل المستقر
        "glm-4-plus",      # GLM (ZhipuAI) - الموديل الأقوى
        "qwen-3-32b",      # Cerebras
    ]
    
    selected_model = st.radio("اختر موديل للتجربة:", models)
    
    if st.button("🗑️ مسح الذاكرة"):
        st.session_state.messages = []
        st.rerun()

# --- 4. الدالة الذكية (تختار الرابط والمفتاح حسب الموديل) ---
def stream_chat_debug(messages, selected_model, c_key, g_key):
    
    # تحديد الإعدادات بناءً على اسم الموديل
    if "glm" in selected_model.lower():
        # إعدادات GLM (ZhipuAI)
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        api_key = g_key
        if not api_key:
            yield "⛔ **خطأ:** لم يتم إدخال مفتاح GLM."
            return
    else:
        # إعدادات Cerebras الافتراضية
        url = "https://api.cerebras.ai/v1/chat/completions"
        api_key = c_key
        if not api_key:
            yield "⛔ **خطأ:** لم يتم إدخال مفتاح Cerebras."
            return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": selected_model,
        "messages": messages,
        "stream": True,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        
        # إذا كان هناك خطأ من السيرفر
        if response.status_code != 200:
            error_details = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get('error', {}).get('message', error_details)
                yield f"⛔ **فشل الموديل:** {selected_model}\n\n**السبب:** {error_msg}"
            except:
                yield f"⛔ **خطأ غير معروف:** رمز الحالة {response.status_code}\n{error_details}"
            return

        # معالجة البث (Streaming)
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8').replace("data: ", "")
                if decoded.strip() == "[DONE]": break
                try:
                    chunk = json.loads(decoded)
                    # GLM و Cerebras يشتركان في نفس هيكلية الرد تقريباً (OpenAI Compatible)
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

if prompt := st.chat_input("اكتب رسالتك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_holder = st.empty()
        full_text = ""
        
        # تمرير المفاتيح والموديل المختار للدالة
        stream_gen = stream_chat_debug(
            st.session_state.messages, 
            selected_model, 
            cerebras_key, 
            glm_key
        )
        
        for chunk in stream_gen:
            full_text += chunk
            response_holder.markdown(full_text + "▌")
        
        response_holder.markdown(full_text)
        
        if "⛔" not in full_text:
            st.session_state.messages.append({"role": "assistant", "content": full_text})
