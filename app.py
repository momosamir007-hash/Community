"""
🤖 تطبيق Streamlit لاستخدام GLM API
جميع النماذج المتاحة من Zhipu AI
"""

import streamlit as st
from openai import OpenAI
import time

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="GLM Chat - محادثة ذكية",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== النماذج المتاحة ====================
GLM_MODELS = {
    "glm-4-plus": {
        "name": "GLM-4 Plus ⭐",
        "description": "الأحدث والأقوى - أداء متفوق",
        "max_tokens": 128000,
        "recommended": True
    },
    "glm-4": {
        "name": "GLM-4",
        "description": "نموذج متعدد الاستخدامات",
        "max_tokens": 128000,
        "recommended": False
    },
    "glm-4-air": {
        "name": "GLM-4 Air 🚀",
        "description": "سريع وفعال للمهام اليومية",
        "max_tokens": 128000,
        "recommended": False
    },
    "glm-4-flash": {
        "name": "GLM-4 Flash ⚡",
        "description": "الأسرع - مثالي للردود السريعة",
        "max_tokens": 128000,
        "recommended": False
    },
    "glm-4-long": {
        "name": "GLM-4 Long 📚",
        "description": "للنصوص الطويلة والوثائق",
        "max_tokens": 1024000,
        "recommended": False
    },
    "glm-3-turbo": {
        "name": "GLM-3 Turbo",
        "description": "نموذج الجيل السابق - اقتصادي",
        "max_tokens": 32000,
        "recommended": False
    }
}

# ==================== إعدادات الشريط الجانبي ====================
with st.sidebar:
    st.title("⚙️ إعدادات")
    
    # API Key
    api_key = st.text_input(
        "🔑 API Key",
        value="f238665f81e44fad90c96cee0220b018.UnH1zIyvieg0zAnj",
        type="password",
        help="أدخل API Key الخاص بك من open.bigmodel.cn"
    )
    
    st.divider()
    
    # اختيار النموذج
    st.subheader("🧠 اختيار النموذج")
    
    # ترتيب النماذج (الموصى بها أولاً)
    sorted_models = sorted(GLM_MODELS.items(), key=lambda x: not x[1]["recommended"])
    
    model_options = [f"{v['name']}" for k, v in sorted_models]
    model_keys = [k for k, v in sorted_models]
    
    selected_model_index = st.selectbox(
        "اختر النموذج:",
        range(len(model_options)),
        format_func=lambda i: model_options[i]
    )
    selected_model = model_keys[selected_model_index]
    model_info = GLM_MODELS[selected_model]
    
    st.caption(f"📝 {model_info['description']}")
    st.caption(f"📊 الحد الأقصى: {model_info['max_tokens']:,} tokens")
    
    st.divider()
    
    # إعدادات متقدمة
    st.subheader("🎛️ إعدادات متقدمة")
    
    with st.expander("🔧 تخصيص المعاملات", expanded=False):
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="قيم أعلى = إجابات أكثر إبداعاً"
        )
        
        top_p = st.slider(
            "🎯 Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="تنويع الإجابات"
        )
        
        max_tokens = st.slider(
            "📏 Max Tokens",
            min_value=100,
            max_value=min(4096, model_info["max_tokens"]),
            value=2048,
            step=100,
            help="الحد الأقصى لطول الرد"
        )
        
        stream_response = st.checkbox(
            "🌊 Stream Mode",
            value=True,
            help="عرض الرد تدريجياً"
        )
    
    st.divider()
    
    # System Prompt
    st.subheader("💬 System Prompt")
    system_prompt = st.text_area(
        "تعليمات النظام:",
        value="أنت مساعد ذكي ومفيد. أجب باللغة العربية إلا إذا طُلب منك غير ذلك.",
        height=100
    )
    
    st.divider()
    
    # أزرار التحكم
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 إعادة تعيين", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # معلومات
    st.divider()
    st.markdown("""
    ### 📖 معلومات
    
    **GLM** هي نماذج ذكاء اصطناعي من **Zhipu AI**
    
    🔗 [open.bigmodel.cn](https://open.bigmodel.cn)
    
    ---
    *تم التطوير بواسطة GLM API*
    """)

# ==================== الوظائف المساعدة ====================

def get_client(api_key: str) -> OpenAI:
    """إنشاء عميل OpenAI متوافق مع GLM"""
    return OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

def stream_chat(client: OpenAI, messages: list, model: str, **kwargs):
    """بث الرد تدريجياً"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9),
        max_tokens=kwargs.get("max_tokens", 2048),
        stream=True
    )
    return response

def normal_chat(client: OpenAI, messages: list, model: str, **kwargs):
    """الحصول على الرد كاملاً"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9),
        max_tokens=kwargs.get("max_tokens", 2048),
        stream=False
    )
    return response

# ==================== تهيئة المحادثة ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== واجهة المحادثة ====================
st.title("🤖 GLM Chat - محادثة ذكية")
st.caption(f"النموذج الحالي: **{model_info['name']}** | {model_info['description']}")

# عرض الرسائل السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# حقل الإدخال
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    # التحقق من API Key
    if not api_key:
        st.error("❌ الرجاء إدخال API Key")
    else:
        # إضافة رسالة المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # إعداد الرسائل للإرسال
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(st.session_state.messages)
        
        # الحصول على الرد
        with st.chat_message("assistant"):
            try:
                client = get_client(api_key)
                
                if stream_response:
                    # وضع البث
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    response = stream_chat(
                        client, messages, selected_model,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens
                    )
                    
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                else:
                    # الوضع العادي
                    with st.spinner("جاري التفكير..."):
                        response = normal_chat(
                            client, messages, selected_model,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens
                        )
                    
                    full_response = response.choices[0].message.content
                    st.markdown(full_response)
                
                # إضافة الرد للمحادثة
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"❌ حدث خطأ: {str(e)}")
                if "401" in str(e):
                    st.warning("⚠️ تحقق من صحة API Key")
                elif "429" in str(e):
                    st.warning("⚠️ تم تجاوز حد الطلبات، الرجاء المحاولة لاحقاً")

# ==================== تذييل ====================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📊 عدد الرسائل: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")
with col2:
    st.caption(f"🧠 النموذج: {selected_model}")
with col3:
    st.caption("💎 Powered by GLM API")
