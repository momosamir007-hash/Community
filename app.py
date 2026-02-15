import streamlit as st
import requests
import json

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="AI Debugger (Cerebras + GLM-5)", page_icon="🧪")

# تخصيص CSS
st.markdown("""
<style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stTextInput > div > div > input { direction: rtl; text-align: right; }
    .stSelectbox > div > div > div { direction: rtl; }
    .stExpander { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 فحص موديلات (Cerebras + GLM-5)")

# --- 2. إعداد المفاتيح والروابط ---
with st.sidebar:
    st.header("🔑 إعدادات المفاتيح")
    
    # 1. مفتاح Cerebras
    try:
        cerebras_key = st.secrets["CEREBRAS_API_KEY"]
    except:
        cerebras_key = st.text_input("مفتاح Cerebras API:", type="password")

    # 2. مفتاح Zed.ai / GLM
    try:
        zed_key = st.secrets["ZED_API_KEY"]
    except:
        zed_key = st.text_input("مفتاح Zed.ai API:", type="password")

    st.markdown("---")
    
    # إعدادات الروابط
    with st.expander("⚙️ إعدادات الروابط (Base URLs)"):
        # ملاحظة: إذا كنت تستخدم chat.z.ai، قد يكون الرابط مختلفاً عن الرابط الرسمي
        # الرابط الرسمي هو: https://open.bigmodel.cn/api/paas/v4/chat/completions
        # رابط chat.z.ai المتوقع (جرب هذا إذا لم يعمل الرسمي): https://chat.z.ai/api/v1/chat/completions
        
        default_zed_url = st.text_input(
            "رابط Zed.ai / GLM:", 
            value="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            help="إذا لم يعمل، جرب: https://chat.z.ai/api/v1/chat/completions"
        )
        
        cerebras_url = "https://api.cerebras.ai/v1/chat/completions"

# --- 3. اختيار الموديل (تحديث القائمة حسب الصورة) ---
with st.sidebar:
    st.header("🤖 اختيار الموديل")
    
    model_options = {
        "Cerebras": [
            "llama-3.3-70b",
            "llama3.1-8b",
            "qwen-3-32b"
        ],
        "Zed.ai (GLM)": [
            "glm-5",           # ✅ الجديد (Flagship)
            "glm-4.7",         # ✅ موديل قوي
            "glm-4.6",         # ✅ كلاسيكي عالي الأداء
            "glm-4-plus",      # القديم القوي
            "glm-4-air",       # سريع
            "glm-4-flash"      # اقتصادي
        ]
    }
    
    provider = st.selectbox("المزود:", list(model_options.keys()))
    
    # خيار لإدخال اسم موديل يدوياً في حالة ظهور موديلات جديدة
    selected_model_dropdown = st.selectbox("الموديل:", model_options[provider])
    use_manual = st.checkbox("كتابة اسم الموديل يدوياً؟")
    
    if use_manual:
        selected_model = st.text_input("اكتب اسم الموديل (مثال: glm-4.6v):", value=selected_model_dropdown)
    else:
        selected_model = selected_model_dropdown

    if st.button("🗑️ مسح الذاكرة"):
        st.session_state.messages = []
        st.rerun()

# --- 4. دالة الاتصال الموحدة ---
def stream_chat_debug(messages, model, provider_name, c_key, z_key, c_url, z_url):
    
    # تحديد الإعدادات
    if provider_name == "Zed.ai (GLM)":
        url = z_url
        api_key = z_key
        if not api_key:
            yield "⛔ **خطأ:** الرجاء إدخال مفتاح Zed.ai."
            return
    else:
        url = c_url
        api_key = c_key
        if not api_key:
            yield "⛔ **خطأ:** الرجاء إدخال مفتاح Cerebras."
            return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 1500 
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        
        if response.status_code != 200:
            try:
                err_json = response.json()
                err_msg = err_json.get('error', {}).get('message', response.text)
                yield f"⛔ **فشل الاتصال بـ {provider_name}:**\nرمز الخطأ: {response.status_code}\nالرسالة: {err_msg}"
            except:
                yield f"⛔ **خطأ غير معروف:** {response.text}"
            return

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
        yield f"❌ **خطأ في الشبكة:** {e}"

# --- 5. واجهة المحادثة ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اكتب رسالتك..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_holder = st.empty()
        full_text = ""
        
        stream_gen = stream_chat_debug(
            st.session_state.messages, 
            selected_model, 
            provider,
            cerebras_key, 
            zed_key,
            cerebras_url,
            default_zed_url  # استخدام المتغير الصحيح
        )
        
        for chunk in stream_gen:
            full_text += chunk
            response_holder.markdown(full_text + "▌")
        
        response_holder.markdown(full_text)
        
        if "⛔" not in full_text and "❌" not in full_text:
            st.session_state.messages.append({"role": "assistant", "content": full_text})

