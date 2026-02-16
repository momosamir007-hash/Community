import streamlit as st
import requests
import json
import time
import sqlite3
import pandas as pd
import plotly.express as px
import re
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ==========================================
# 1. إعدادات الصفحة (يجب أن تكون في البداية)
# ==========================================
st.set_page_config(
    page_title="CineMate Pro - الناقد السينمائي",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. التخصيص والمظهر (CSS)
# ==========================================
def apply_theme(theme):
    if theme == "داكن":
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#1e2130"
    else:
        bg_color = "#ffffff"
        text_color = "#31333F"
        card_bg = "#f0f2f6"
    
    st.markdown(f"""
    <style>
        .main {{direction: rtl; text-align: right; background-color: {bg_color}; color: {text_color};}}
        .stTextInput > div > div > input {{text-align: right; direction: rtl;}}
        .stSelectbox > div > div {{direction: rtl;}}
        h1, h2, h3, p {{font-family: 'Segoe UI', Tahoma, sans-serif;}}
        .metric-card {{background-color: {card_bg}; padding: 15px; border-radius: 10px; border: 1px solid #444; text-align: center;}}
        div[data-testid="stMetricValue"] {{font-size: 1.5rem;}}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. نماذج البيانات (Pydantic Models)
# ==========================================
class MovieInfo(BaseModel):
    arabic_title: str = Field(..., description="عنوان العمل بالعربية")
    original_title: str = Field(..., description="العنوان الأصلي")
    year: int = Field(..., description="سنة الإصدار")
    director: str = Field(..., description="اسم المخرج")
    genre: List[str] = Field(..., description="قائمة التصنيفات بالعربية")
    type: str = Field("فيلم", description="فيلم أو مسلسل")

class TechnicalAnalysis(BaseModel):
    screenplay: str = Field(..., description="تحليل عميق للقصة والسيناريو")
    acting: str = Field(..., description="تحليل الأداء التمثيلي")
    visuals: str = Field(..., description="الإخراج، التصوير، والإضاءة")
    music: str = Field(..., description="الموسيقى والصوتيات")
    symbolism: str = Field(..., description="الرسائل الضمنية والرمزية")

class Recommendation(BaseModel):
    score: float = Field(..., description="التقييم من 10")
    pros: List[str] = Field(..., description="أبرز 3 إيجابيات")
    cons: List[str] = Field(..., description="أبرز 3 سلبيات")
    similar_movies: List[str] = Field(..., description="3 أعمال مشابهة")
    streaming_on: List[str] = Field(..., description="منصات المشاهدة")
    final_verdict: str = Field(..., description="حكم نهائي مختصر ومحترف")

class FullMovieReport(BaseModel):
    info: MovieInfo
    analysis: TechnicalAnalysis
    recommendation: Recommendation

# ==========================================
# 4. قاعدة البيانات (SQLite)
# ==========================================
def init_db():
    try:
        conn = sqlite3.connect('cinemate_v3.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reports
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      title TEXT,
                      arabic_title TEXT,
                      director TEXT,
                      score REAL,
                      year INTEGER,
                      type TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"خطأ في قاعدة البيانات: {e}")

def save_report_to_db(report: FullMovieReport):
    try:
        conn = sqlite3.connect('cinemate_v3.db')
        c = conn.cursor()
        # تحويل القوائم إلى نصوص JSON بسيطة للتخزين
        c.execute('''INSERT INTO reports (title, arabic_title, director, score, year, type)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (report.info.original_title,
                   report.info.arabic_title,
                   report.info.director,
                   report.recommendation.score,
                   report.info.year,
                   report.info.type))
        conn.commit()
        conn.close()
    except Exception as e:
        pass # تجاهل أخطاء التخزين لعدم إيقاف التطبيق

def get_reports_from_db(limit=10):
    try:
        conn = sqlite3.connect('cinemate_v3.db')
        df = pd.read_sql_query("SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))
        conn.close()
        return df
    except:
        return pd.DataFrame()

init_db()

# ==========================================
# 5. دوال TMDB (بيانات الأفلام والصور)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tmdb_data(api_key: str, query: str, is_tv: bool = False):
    """جلب بيانات وصور من TMDB"""
    if not api_key:
        return None
    
    base_url = "https://api.themoviedb.org/3"
    endpoint = "search/tv" if is_tv else "search/movie"
    
    try:
        params = {"api_key": api_key, "query": query, "language": "ar-SA"}
        response = requests.get(f"{base_url}/{endpoint}", params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if not data['results']:
            return None
            
        item = data['results'][0]
        item_id = item['id']
        
        # جلب التفاصيل الإضافية (يوتيوب + ممثلين)
        type_path = "tv" if is_tv else "movie"
        details_url = f"{base_url}/{type_path}/{item_id}"
        details_params = {"api_key": api_key, "append_to_response": "credits,videos,recommendations", "language": "ar-SA"}
        
        details_resp = requests.get(details_url, params=details_params, timeout=5)
        details_data = details_resp.json()
        
        # استخراج البيانات
        cast = [p['name'] for p in details_data.get('credits', {}).get('cast', [])[:5]]
        
        trailer_key = None
        for vid in details_data.get('videos', {}).get('results', []):
            if vid['site'] == 'YouTube' and vid['type'] == 'Trailer':
                trailer_key = vid['key']
                break
        
        similar = [s['name'] if is_tv else s['title'] for s in details_data.get('recommendations', {}).get('results', [])[:3]]
        
        return {
            'poster': f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None,
            'backdrop': f"https://image.tmdb.org/t/p/w1280{item.get('backdrop_path')}" if item.get('backdrop_path') else None,
            'rating': item.get('vote_average'),
            'overview': item.get('overview'),
            'cast': cast,
            'trailer_key': trailer_key,
            'similar_tmdb': similar
        }
    except Exception:
        return None

# ==========================================
# 6. محرك الذكاء الاصطناعي (Gemini 1.5 Pro/Flash) - مصحح
# ==========================================
def clean_json_text(text):
    """تنظيف النص لإزالة markdown json wrappers"""
    # حذف ```json في البداية
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    # حذف ``` في النهاية
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()

def call_gemini_smart(api_key, prompt):
    """استدعاء Gemini مع التبديل الذكي بين الموديلات"""
    genai.configure(api_key=api_key)
    
    # القائمة: نبدأ بالأقوى (Pro) ثم الأسرع (Flash)
    # ملاحظة: gemini-1.5-pro-latest قد لا يعمل دائماً، نستخدم الاسم المستقر
    models_to_try = ["gemini-1.5-pro", "gemini-1.5-flash"]
    
    last_exception = None

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json" # ميزة مهمة جداً لضمان JSON
                }
            )
            
            # محاولة التوليد
            response = model.generate_content(prompt)
            
            if not response.parts:
                raise ValueError("استجابة فارغة (Blocked)")
                
            text_response = response.text
            cleaned_text = clean_json_text(text_response)
            
            # محاولة تحويل النص إلى JSON
            return json.loads(cleaned_text)
            
        except Exception as e:
            last_exception = e
            # استمر للموديل التالي في القائمة
            continue
    
    # إذا فشلت كل الموديلات
    st.error(f"فشل الاتصال بجميع نماذج Gemini. الخطأ الأخير: {last_exception}")
    return None

def analyze_movie(api_key: str, movie_name: str, content_type: str = "فيلم", comparison_mode: bool = False, other_movies: List[str] = None):
    """تجهيز الطلب وإرساله"""
    
    # تحضير Schema
    schema_str = json.dumps(FullMovieReport.model_json_schema(), indent=2, ensure_ascii=False)
    
    if comparison_mode and other_movies:
        all_movies = [movie_name] + other_movies
        movies_str = "، ".join(all_movies)
        
        prompt = f"""
        You are an elite Arab Film Critic. Compare these {content_type}s: {movies_str}.
        
        Task:
        1. Analyze EACH movie separately using the schema below.
        2. Provide a comparison summary.
        
        Output JSON Structure:
        {{
            "movies": [List of FullMovieReport objects],
            "comparison": {{
                "better_plot": "Movie Title",
                "better_acting": "Movie Title",
                "better_visuals": "Movie Title",
                "better_music": "Movie Title",
                "overall_winner": "Movie Title",
                "verdict": "Detailed Arabic comparison verdict"
            }}
        }}
        
        Schema for 'FullMovieReport':
        {schema_str}
        
        Language: Arabic (Fusha). strictly JSON.
        """
    else:
        prompt = f"""
        Act as a professional Arab Film Critic. Analyze the {content_type}: "{movie_name}".
        
        Return STRICT JSON matching this schema:
        {schema_str}
        
        Language: Arabic (Fusha). Ensure valid JSON.
        """
    
    return call_gemini_smart(api_key, prompt)

# ==========================================
# 7. الواجهة الرئيسية
# ==========================================
def main():
    # --- الشريط الجانبي ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=80)
        st.title("الإعدادات")
        
        # إدارة المفاتيح
        gemini_key = st.session_state.get('gemini_key', '')
        tmdb_key = st.session_state.get('tmdb_key', '')
        
        # محاولة القراءة من secrets
        if not gemini_key and "gemini_key" in st.secrets:
            gemini_key = st.secrets["gemini_key"]
        if not tmdb_key and "tmdb_key" in st.secrets:
            tmdb_key = st.secrets["tmdb_key"]
            
        new_g_key = st.text_input("مفتاح Gemini API", value=gemini_key, type="password")
        new_t_key = st.text_input("مفتاح TMDB API (اختياري)", value=tmdb_key, type="password")
        
        if new_g_key: st.session_state['gemini_key'] = new_g_key
        if new_t_key: st.session_state['tmdb_key'] = new_t_key
        
        st.divider()
        
        theme = st.selectbox("المظهر", ["فاتح", "داكن"])
        apply_theme(theme)
        
        content_type = st.radio("نوع المحتوى", ["فيلم", "مسلسل"], horizontal=True)
        comparison_mode = st.checkbox("وضع المقارنة")
        
        num_comp = 2
        if comparison_mode:
            num_comp = st.number_input("عدد الأعمال", 2, 4, 2)
            
        st.divider()
        st.caption("سجل البحث:")
        hist = get_reports_from_db(5)
        if not hist.empty:
            for _, r in hist.iterrows():
                st.text(f"▫️ {r['arabic_title']} ({r['score']})")

    # --- المحتوى الرئيسي ---
    st.title("🎬 CineMate Pro")
    st.subheader("الناقد السينمائي الذكي (Gemini 1.5 Pro)")
    
    if not st.session_state.get('gemini_key'):
        st.warning("⚠️ الرجاء إدخال مفتاح Gemini API في القائمة الجانبية.")
        st.stop()
        
    # حقول الإدخال
    inputs = []
    cols = st.columns(num_comp if comparison_mode else 1)
    for i, col in enumerate(cols):
        with col:
            val = st.text_input(f"العمل {i+1}", key=f"in_{i}", placeholder="مثال: The Godfather")
            if val: inputs.append(val)
            
    if st.button("🚀 تحليل الآن", use_container_width=True):
        if not inputs:
            st.error("الرجاء إدخال اسم العمل الفني.")
        else:
            bar = st.progress(0, "جاري التحضير...")
            
            # 1. جلب بيانات TMDB (توازي)
            tmdb_results = []
            if st.session_state.get('tmdb_key'):
                for idx, mov in enumerate(inputs):
                    bar.progress((idx+1)*10, f"جلب صور {mov}...")
                    t_data = fetch_tmdb_data(st.session_state['tmdb_key'], mov, content_type=="مسلسل")
                    tmdb_results.append(t_data)
            
            # 2. تحليل Gemini
            bar.progress(50, "جاري التحليل العميق باستخدام Gemini 1.5 Pro...")
            
            try:
                result = analyze_movie(
                    st.session_state['gemini_key'], 
                    inputs[0], 
                    content_type, 
                    comparison_mode, 
                    inputs[1:] if comparison_mode else None
                )
                
                bar.progress(100, "تم!")
                time.sleep(0.5)
                bar.empty()
                
                if result:
                    # توحيد الهيكل
                    reports = []
                    comp_data = None
                    
                    if comparison_mode and isinstance(result, dict) and 'movies' in result:
                        # تحويل الـ dict إلى Objects
                        reports = [FullMovieReport(**m) for m in result['movies']]
                        comp_data = result.get('comparison')
                    elif isinstance(result, FullMovieReport):
                        reports = [result]
                    elif isinstance(result, dict):
                        # حالة فردية ولكن عادت كـ dict
                        reports = [FullMovieReport(**result)]

                    # --- عرض النتائج ---
                    
                    # قسم المقارنة
                    if comp_data:
                        st.header("⚖️ ملخص المقارنة")
                        col_w, col_v = st.columns([1, 2])
                        col_w.metric("🏆 الفائز", comp_data.get('overall_winner', 'N/A'))
                        col_v.info(comp_data.get('verdict', ''))
                        
                        comp_df = pd.DataFrame({
                            "المعيار": ["القصة", "التمثيل", "البصريات", "الموسيقى"],
                            "الأفضل": [
                                comp_data.get('better_plot'),
                                comp_data.get('better_acting'),
                                comp_data.get('better_visuals'),
                                comp_data.get('better_music')
                            ]
                        })
                        st.table(comp_df)
                        
                        # رسم بياني
                        scores = {r.info.arabic_title: r.recommendation.score for r in reports}
                        fig = px.bar(x=list(scores.keys()), y=list(scores.values()), title="مقارنة التقييمات", labels={'y':'التقييم', 'x':'العمل'})
                        st.plotly_chart(fig, use_container_width=True)
                        st.divider()

                    # عرض التقارير الفردية
                    for idx, report in enumerate(reports):
                        # حفظ في DB
                        save_report_to_db(report)
                        
                        # ربط مع بيانات TMDB
                        t_data = tmdb_results[idx] if idx < len(tmdb_results) else None
                        
                        with st.container():
                            # Header
                            c_img, c_txt = st.columns([1, 4])
                            with c_img:
                                if t_data and t_data.get('poster'):
                                    st.image(t_data['poster'], use_container_width=True)
                                else:
                                    st.markdown("🖼️ لا توجد صورة")
                            
                            with c_txt:
                                st.subheader(f"{report.info.arabic_title} ({report.info.year})")
                                st.caption(f"{report.info.original_title} | {report.info.director}")
                                
                                m1, m2, m3 = st.columns(3)
                                m1.metric("التقييم النقدي", f"{report.recommendation.score}/10")
                                m2.metric("النوع", ", ".join(report.info.genre[:2]))
                                if t_data:
                                    m3.metric("تقييم الجمهور", f"{t_data.get('rating', 'N/A')}")
                                
                                if t_data and t_data.get('trailer_key'):
                                    st.video(f"https://www.youtube.com/watch?v={t_data['trailer_key']}")

                            # Tabs
                            tab1, tab2, tab3 = st.tabs(["التحليل الفني", "المميزات والعيوب", "توصيات"])
                            
                            with tab1:
                                st.markdown(f"**📖 السيناريو:** {report.analysis.screenplay}")
                                st.markdown(f"**🎭 التمثيل:** {report.analysis.acting}")
                                st.markdown(f"**🎥 البصريات:** {report.analysis.visuals}")
                                st.markdown(f"**🎼 الموسيقى:** {report.analysis.music}")
                                st.info(f"💡 **الرمزية:** {report.analysis.symbolism}")
                                
                            with tab2:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.success("✅ نقاط القوة")
                                    for p in report.recommendation.pros: st.write(f"- {p}")
                                with c2:
                                    st.error("❌ نقاط الضعف")
                                    for c in report.recommendation.cons: st.write(f"- {c}")
                                st.markdown(f"**الحكم النهائي:** {report.recommendation.final_verdict}")
                                
                            with tab3:
                                st.write(f"📺 **منصات:** {', '.join(report.recommendation.streaming_on)}")
                                st.write(f"🔗 **مشابه (AI):** {', '.join(report.recommendation.similar_movies)}")
                                if t_data and t_data.get('similar_tmdb'):
                                    st.write(f"🌍 **مشابه (TMDB):** {', '.join(t_data['similar_tmdb'])}")
                        
                        st.markdown("---")

            except Exception as e:
                st.error(f"حدث خطأ غير متوقع: {str(e)}")

if __name__ == "__main__":
    main()

