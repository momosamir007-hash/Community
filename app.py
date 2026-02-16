import streamlit as st
import requests
import json
import time
from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# 1. إعدادات الصفحة والتصميم
# ==========================================
st.set_page_config(
    page_title="CineMate Pro - الناقد السينمائي",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص CSS للغة العربية
st.markdown("""
<style>
    .main {direction: rtl; text-align: right;}
    .stTextInput > div > div > input {text-align: right;}
    h1, h2, h3, p {font-family: 'Tahoma', sans-serif;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #ddd; text-align: center;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. هيكلية البيانات (The Brain - Pydantic)
# ==========================================
class MovieInfo(BaseModel):
    arabic_title: str = Field(..., description="The movie title in Arabic")
    original_title: str = Field(..., description="The original title")
    year: int = Field(..., description="Release year")
    director: str = Field(..., description="Director name")
    duration: str = Field(..., description="Duration (e.g., 2h 15m)")
    genre: List[str] = Field(..., description="List of genres in Arabic")

class TechnicalAnalysis(BaseModel):
    screenplay: str = Field(..., description="Deep analysis of the plot and writing in Arabic")
    acting: str = Field(..., description="Analysis of acting performances in Arabic")
    visuals: str = Field(..., description="Cinematography, lighting, and directing style in Arabic")
    music: str = Field(..., description="Soundtrack and sound design analysis in Arabic")
    symbolism: str = Field(..., description="Hidden themes and philosophical messages in Arabic")

class Recommendation(BaseModel):
    score: float = Field(..., description="Score out of 10")
    pros: List[str] = Field(..., description="Top 3 pros")
    cons: List[str] = Field(..., description="Top 3 cons")
    similar_movies: List[str] = Field(..., description="3 similar movies titles")
    streaming_on: List[str] = Field(..., description="Where to watch (Netflix, etc.)")
    final_verdict: str = Field(..., description="A short, professional final verdict in Arabic")

class FullMovieReport(BaseModel):
    info: MovieInfo
    analysis: TechnicalAnalysis
    recommendation: Recommendation

# ==========================================
# 3. محرك التحليل (Cerebras Engine)
# ==========================================
def analyze_movie(api_key: str, movie_name: str) -> Optional[FullMovieReport]:
    """
    يتصل بـ Cerebras API ويحلل الفيلم ويعيد كائن Pydantic
    """
    API_URL = "https://api.cerebras.ai/v1/chat/completions"
    MODEL = "llama-3.3-70b"

    # تجهيز مخطط JSON للهيكلية
    schema_json = json.dumps(FullMovieReport.model_json_schema(), indent=2)

    messages = [
        {
            "role": "system",
            "content": f"""
            You are an elite Arab Film Critic (like Youssef Chahine mixed with Roger Ebert).
            Analyze the requested movie/series deeply.
            Language: High-quality Arabic (Fusha).
            You MUST output strict JSON following this schema:
            {schema_json}
            """
        },
        {
            "role": "user",
            "content": f"Analyze: {movie_name}"
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status() # التأكد من عدم وجود أخطاء HTTP
        
        data = response.json()
        content = data['choices'][0]['message']['content']
        
        # تحويل النص إلى كائن بايثون والتحقق منه
        parsed_data = json.loads(content)
        return FullMovieReport(**parsed_data)

    except Exception as e:
        st.error(f"حدث خطأ أثناء الاتصال: {str(e)}")
        if 'response' in locals():
            st.code(response.text) # عرض الخطأ الخام للمساعدة
        return None

# ==========================================
# 4. واجهة التطبيق (Frontend Logic)
# ==========================================

# --- الشريط الجانبي ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=100)
    st.title("إعدادات المحرك")
    
    # إدخال المفتاح (مع حفظه في الجلسة)
    api_key_input = st.text_input("مفتاح Cerebras API", type="password", help="يبدأ بـ csk-")
    if api_key_input:
        st.session_state['api_key'] = api_key_input
    
    st.info("💡 هذا المشروع يستخدم Llama-3.3-70b عبر Cerebras لسرعة فائقة.")
    st.markdown("---")
    st.write("Designed by: **AI Architect**")

# --- الواجهة الرئيسية ---
st.title("🎬 CineMate Pro")
st.subheader("منصة التحليل السينمائي المتقدمة")

# التحقق من المفتاح
if 'api_key' not in st.session_state:
    st.warning("⚠️ يرجى إدخال مفتاح API في القائمة الجانبية للبدء.")
    st.stop()

# مربع البحث
col1, col2 = st.columns([3, 1])
with col1:
    movie_name = st.text_input("اسم الفيلم أو المسلسل:", placeholder="مثال: The Godfather, Interstellar...")
with col2:
    st.write("") # مسافة
    st.write("") 
    analyze_btn = st.button("🔍 تحليل شامل", use_container_width=True)

# منطق العرض
if analyze_btn and movie_name:
    with st.spinner(f"جاري استحضار النقد السينمائي لـ '{movie_name}'..."):
        report = analyze_movie(st.session_state['api_key'], movie_name)
        
        if report:
            # --- رأس الصفحة (Info Header) ---
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("العنوان", report.info.arabic_title)
            c2.metric("السنة", report.info.year)
            c3.metric("المخرج", report.info.director)
            c4.metric("التقييم", f"{report.recommendation.score}/10")
            
            # --- التصنيفات (Tags) ---
            st.write("**التصنيف:** " + ", ".join([f"`{g}`" for g in report.info.genre]))
            
            # --- المحتوى الرئيسي (Tabs) ---
            tab1, tab2, tab3 = st.tabs(["📝 التحليل الفني", "⚖️ الحكم والمميزات", "🧠 العمق والرسائل"])
            
            with tab1:
                st.header("التحليل الفني")
                
                st.subheader("📖 السيناريو والحبكة")
                st.write(report.analysis.screenplay)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("🎭 الأداء التمثيلي")
                    st.info(report.analysis.acting)
                with col_b:
                    st.subheader("🎥 الإخراج والبصريات")
                    st.success(report.analysis.visuals)
                
                st.subheader("🎼 الموسيقى والصوت")
                st.write(report.analysis.music)

            with tab2:
                st.header("الحكم النهائي")
                
                c_pros, c_cons = st.columns(2)
                with c_pros:
                    st.success("✅ **نقاط القوة:**")
                    for p in report.recommendation.pros:
                        st.write(f"- {p}")
                
                with c_cons:
                    st.error("❌ **نقاط الضعف:**")
                    for c in report.recommendation.cons:
                        st.write(f"- {c}")
                
                st.markdown("---")
                st.subheader("💡 الحكم:")
                st.warning(f"**{report.recommendation.final_verdict}**")
                
                st.write("**📺 متوفر على:** " + ", ".join(report.recommendation.streaming_on))
                st.write("**🤔 أفلام مشابهة:** " + ", ".join(report.recommendation.similar_movies))

            with tab3:
                st.header("ما وراء الصورة")
                st.markdown(f"> {report.analysis.symbolism}")
                
                # تصور بياني بسيط (Dummy Visual)
                st.progress(report.recommendation.score / 10, text="جودة العمل الفني")

else:
    if not movie_name and analyze_btn:
        st.error("الرجاء كتابة اسم الفيلم أولاً.")
