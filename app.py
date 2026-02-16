import streamlit as st
import requests
import json
import time
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from collections import Counter
from datetime import datetime
import hashlib
import re
import google.generativeai as genai

# ==========================================
# 1. إعدادات الصفحة والتصميم مع الوضع الليلي
# ==========================================
st.set_page_config(
    page_title="CineMate Pro - الناقد السينمائي",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص CSS للغة العربية والوضع الليلي
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
        .stTextInput > div > div > input {{text-align: right;}}
        h1, h2, h3, p {{font-family: 'Tahoma', sans-serif;}}
        .metric-card {{background-color: {card_bg}; padding: 15px; border-radius: 10px; border: 1px solid #ddd; text-align: center;}}
        .tmdb-card {{background-color: #0e1a2b; color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px;}}
        .comparison-table {{background-color: {card_bg}; border-radius: 10px; padding: 10px;}}
        .stButton>button {{width: 100%;}}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. هيكلية البيانات (Pydantic)
# ==========================================
class MovieInfo(BaseModel):
    arabic_title: str = Field(..., description="The movie title in Arabic")
    original_title: str = Field(..., description="The original title")
    year: int = Field(..., description="Release year")
    director: str = Field(..., description="Director name")
    duration: str = Field(..., description="Duration (e.g., 2h 15m)")
    genre: List[str] = Field(..., description="List of genres in Arabic")
    type: str = Field("فيلم", description="فيلم أو مسلسل")

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
# 3. قاعدة بيانات محلية (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('cinemate.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  arabic_title TEXT,
                  director TEXT,
                  genres TEXT,
                  score REAL,
                  year INTEGER,
                  type TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_report_to_db(report: FullMovieReport):
    conn = sqlite3.connect('cinemate.db')
    c = conn.cursor()
    c.execute('''INSERT INTO reports (title, arabic_title, director, genres, score, year, type)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (report.info.original_title,
               report.info.arabic_title,
               report.info.director,
               json.dumps(report.info.genre),
               report.recommendation.score,
               report.info.year,
               report.info.type))
    conn.commit()
    conn.close()

def get_reports_from_db(limit=50):
    conn = sqlite3.connect('cinemate.db')
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    # تحويل genres من JSON إلى قائمة
    if not df.empty:
        df['genres'] = df['genres'].apply(json.loads)
    return df

# تهيئة قاعدة البيانات
init_db()

# ==========================================
# 4. دوال مساعدة (TMDB Integration مع تخزين مؤقت)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tmdb_data(api_key: str, movie_name: str):
    """جلب بيانات إضافية من TMDB مع تخزين مؤقت"""
    if not api_key:
        return None
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": api_key,
            "query": movie_name,
            "language": "ar-SA"
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            movie_id = movie['id']
            
            # جلب تفاصيل إضافية (الممثلين، المخرج، الفيديو)
            credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
            credits_response = requests.get(credits_url, params={"api_key": api_key})
            credits_response.raise_for_status()
            credits = credits_response.json()
            
            # جلب الفيديو (trailer)
            videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
            videos_response = requests.get(videos_url, params={"api_key": api_key})
            videos_response.raise_for_status()
            videos = videos_response.json()
            
            trailer_key = None
            for vid in videos.get('results', []):
                if vid['type'] == 'Trailer' and vid['site'] == 'YouTube':
                    trailer_key = vid['key']
                    break
            
            # استخراج أبرز 5 ممثلين
            cast = [actor['name'] for actor in credits.get('cast', [])[:5]]
            
            # استخراج المخرج
            director = next((crew['name'] for crew in credits.get('crew', []) if crew['job'] == 'Director'), None)
            
            # جلب توصيات الأفلام المشابهة من TMDB
            recommendations_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations"
            recommendations_response = requests.get(recommendations_url, params={"api_key": api_key})
            recommendations_response.raise_for_status()
            recommendations_data = recommendations_response.json()
            similar_tmdb = [rec['title'] for rec in recommendations_data.get('results', [])[:3]]
            
            return {
                'poster': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
                'rating': movie.get('vote_average'),
                'overview': movie.get('overview'),
                'cast': cast,
                'director': director,
                'trailer_key': trailer_key,
                'similar_tmdb': similar_tmdb,
                'backdrop': f"https://image.tmdb.org/t/p/w1280{movie.get('backdrop_path')}" if movie.get('backdrop_path') else None
            }
    except Exception as e:
        st.warning(f"تعذر جلب بيانات TMDB: {e}")
    return None

@st.cache_data(ttl=3600)
def fetch_tv_data(api_key: str, tv_name: str):
    """جلب بيانات مسلسل من TMDB"""
    if not api_key:
        return None
    try:
        search_url = "https://api.themoviedb.org/3/search/tv"
        params = {
            "api_key": api_key,
            "query": tv_name,
            "language": "ar-SA"
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            tv = data['results'][0]
            tv_id = tv['id']
            
            credits_url = f"https://api.themoviedb.org/3/tv/{tv_id}/credits"
            credits_response = requests.get(credits_url, params={"api_key": api_key})
            credits_response.raise_for_status()
            credits = credits_response.json()
            
            videos_url = f"https://api.themoviedb.org/3/tv/{tv_id}/videos"
            videos_response = requests.get(videos_url, params={"api_key": api_key})
            videos_response.raise_for_status()
            videos = videos_response.json()
            
            trailer_key = None
            for vid in videos.get('results', []):
                if vid['type'] == 'Trailer' and vid['site'] == 'YouTube':
                    trailer_key = vid['key']
                    break
            
            cast = [actor['name'] for actor in credits.get('cast', [])[:5]]
            
            return {
                'poster': f"https://image.tmdb.org/t/p/w500{tv['poster_path']}" if tv.get('poster_path') else None,
                'rating': tv.get('vote_average'),
                'overview': tv.get('overview'),
                'cast': cast,
                'trailer_key': trailer_key,
                'backdrop': f"https://image.tmdb.org/t/p/w1280{tv.get('backdrop_path')}" if tv.get('backdrop_path') else None
            }
    except Exception as e:
        st.warning(f"تعذر جلب بيانات المسلسل: {e}")
    return None

# ==========================================
# 5. محرك التحليل (Gemini) مع إعادة محاولة
# ==========================================
def call_gemini_with_retry(api_key, prompt, max_retries=3, delay=2):
    """استدعاء Gemini API مع إعادة محاولة تلقائية"""
    genai.configure(api_key=api_key)
    
    # اختيار النموذج المناسب (يمكن تغييره إلى pro إذا أردت)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_output_tokens": 4000,
        }
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # التحقق من وجود النص
            if not response.parts:
                raise ValueError("الاستجابة فارغة أو تم حظرها.")
            content = response.text
            
            # محاولة استخراج JSON إذا كان النص مختلطاً
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (attempt + 1))

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_movie(api_key: str, movie_name: str, content_type: str = "فيلم", comparison_mode: bool = False, other_movies: List[str] = None) -> Any:
    """
    يحلل الفيلم أو المسلسل باستخدام Gemini مع دعم المقارنة المتعددة
    """
    schema_json = json.dumps(FullMovieReport.model_json_schema(), indent=2, ensure_ascii=False)
    
    if comparison_mode and other_movies:
        movies_list = [movie_name] + other_movies
        movies_str = "، ".join(movies_list)
        
        prompt = f"""
        You are an elite Arab Film Critic. Compare the following {content_type}s: {movies_str}.
        
        First, analyze each {content_type} separately according to the schema below, then provide a comparison.
        
        The schema for each movie is:
        {schema_json}
        
        Output MUST be a JSON object with keys: 'movies' (list of FullMovieReport for each), and 'comparison' (dict with keys: 'better_plot', 'better_acting', 'better_visuals', 'better_music', 'overall_winner', 'verdict').
        
        Language: Arabic (Fusha).
        """
    else:
        prompt = f"""
        You are an elite Arab Film Critic (like Youssef Chahine mixed with Roger Ebert).
        Analyze the requested {content_type} deeply: {movie_name}
        
        You MUST output strict JSON following this schema:
        {schema_json}
        
        Language: High-quality Arabic (Fusha).
        """
    
    try:
        result = call_gemini_with_retry(api_key, prompt)
        
        if comparison_mode and other_movies:
            if 'movies' in result:
                result['movies'] = [FullMovieReport(**m) for m in result['movies']]
            return result
        else:
            return FullMovieReport(**result)
    except Exception as e:
        st.error(f"حدث خطأ أثناء الاتصال بـ Gemini: {str(e)}")
        return None

# ==========================================
# 6. دوال التوصيات الذكية (محسّنة)
# ==========================================
def get_smart_recommendations(current_report: FullMovieReport, tmdb_data=None, top_n=3):
    """توليد توصيات من TMDB أو من التاريخ المحلي"""
    recommendations = []
    
    # 1. إذا توفرت توصيات TMDB
    if tmdb_data and tmdb_data.get('similar_tmdb'):
        for title in tmdb_data['similar_tmdb'][:top_n]:
            recommendations.append({"title": title, "source": "TMDB"})
    
    # 2. استكمال من قاعدة البيانات المحلية
    if len(recommendations) < top_n:
        df = get_reports_from_db(limit=20)
        if not df.empty:
            current_genres = set(current_report.info.genre)
            current_director = current_report.info.director
            
            # حساب التشابه
            scores = []
            for _, row in df.iterrows():
                if row['title'] == current_report.info.original_title:
                    continue
                genres = set(row['genres'])
                genre_sim = len(current_genres & genres) / max(len(current_genres), len(genres)) if current_genres and genres else 0
                director_match = 1 if current_director == row['director'] else 0
                total_score = genre_sim * 0.6 + director_match * 0.4
                scores.append((total_score, row))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            for score, row in scores[:top_n - len(recommendations)]:
                recommendations.append({
                    "title": row['arabic_title'],
                    "director": row['director'],
                    "score": row['score'],
                    "source": "محلي"
                })
    
    return recommendations

# ==========================================
# 7. دوال التصدير والمشاركة
# ==========================================
def generate_markdown_report(report: FullMovieReport, tmdb_data=None):
    """توليد تقرير بصيغة Markdown"""
    md = f"""
# تقرير فيلم: {report.info.arabic_title} ({report.info.original_title})
**السنة:** {report.info.year} | **المخرج:** {report.info.director} | **التقييم:** {report.recommendation.score}/10
**النوع:** {', '.join(report.info.genre)}

## التحليل الفني
### السيناريو والحبكة
{report.analysis.screenplay}

### الأداء التمثيلي
{report.analysis.acting}

### الإخراج والبصريات
{report.analysis.visuals}

### الموسيقى والصوت
{report.analysis.music}

### الرمزية والعمق
{report.analysis.symbolism}

## الحكم
**نقاط القوة:**
"""
    for p in report.recommendation.pros:
        md += f"- {p}\n"
    md += "**نقاط الضعف:**\n"
    for c in report.recommendation.cons:
        md += f"- {c}\n"
    md += f"""
**الحكم النهائي:** {report.recommendation.final_verdict}
**متوفر على:** {', '.join(report.recommendation.streaming_on)}
**أفلام مشابهة:** {', '.join(report.recommendation.similar_movies)}
"""
    if tmdb_data:
        md += f"\n**تقييم TMDB:** {tmdb_data.get('rating')}/10\n"
        if tmdb_data.get('cast'):
            md += f"**الممثلون:** {', '.join(tmdb_data['cast'])}\n"
    return md

# ==========================================
# 8. واجهة التطبيق الرئيسية
# ==========================================
def main():
    # إعدادات جانبية
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=100)
        st.title("إعدادات المحرك")
        
        # استخدام st.secrets كمصدر رئيسي للمفاتيح
        gemini_key = None
        tmdb_key = None
        
        if "gemini_key" in st.secrets:
            gemini_key = st.secrets["gemini_key"]
            st.success("✅ تم تحميل مفتاح Gemini من الأسرار")
        else:
            gemini_key = st.text_input("مفتاح Gemini API", type="password", help="من Google AI Studio")
        
        if "tmdb_key" in st.secrets:
            tmdb_key = st.secrets["tmdb_key"]
            st.success("✅ تم تحميل مفتاح TMDB من الأسرار")
        else:
            tmdb_key = st.text_input("مفتاح TMDB API (اختياري)", type="password", help="لجلب بيانات إضافية")
        
        # تخزين المفاتيح في الجلسة
        if gemini_key:
            st.session_state['gemini_key'] = gemini_key
        if tmdb_key:
            st.session_state['tmdb_key'] = tmdb_key
        
        # اختيار الوضع (ليلي/نهاري)
        theme = st.selectbox("الوضع", ["فاتح", "داكن"], index=0)
        st.session_state['theme'] = theme
        apply_theme(theme)
        
        # نوع المحتوى
        content_type = st.selectbox("نوع المحتوى", ["فيلم", "مسلسل"], index=0)
        st.session_state['content_type'] = content_type
        
        # وضع المقارنة
        comparison_mode = st.checkbox("🔁 تفعيل وضع المقارنة", value=False)
        st.session_state['comparison_mode'] = comparison_mode
        
        if comparison_mode:
            num_movies = st.number_input("عدد الأفلام للمقارنة", min_value=2, max_value=5, value=2, step=1)
            st.session_state['num_movies'] = num_movies
        
        st.info("💡 يستخدم Gemini 1.5 Flash عبر Google AI.")
        
        # عرض تاريخ التحليلات
        st.markdown("---")
        st.subheader("📜 آخر التحليلات")
        df = get_reports_from_db(5)
        if not df.empty:
            for _, row in df.iterrows():
                st.write(f"**{row['arabic_title']}** - {row['score']}/10")
        
        st.markdown("---")
        st.write("Designed by: **AI Architect**")
    
    # التحقق من المفتاح الأساسي
    if 'gemini_key' not in st.session_state:
        st.warning("⚠️ يرجى إدخال مفتاح Gemini API في القائمة الجانبية للبدء.")
        st.stop()
    
    # الواجهة الرئيسية
    st.title("🎬 CineMate Pro")
    st.subheader("منصة التحليل السينمائي المتقدمة")
    
    # حقول الإدخال حسب الوضع
    movies_list = []
    if st.session_state.get('comparison_mode', False):
        cols = st.columns(st.session_state.get('num_movies', 2))
        for i, col in enumerate(cols):
            with col:
                movie = st.text_input(f"العمل {i+1}:", placeholder=f"مثال: Inception", key=f"movie_{i}")
                if movie:
                    movies_list.append(movie)
        analyze_btn = st.button("🔍 تحليل مقارن شامل", use_container_width=True)
    else:
        movie_name = st.text_input("اسم الفيلم أو المسلسل:", placeholder="مثال: Interstellar", key="single_movie")
        analyze_btn = st.button("🔍 تحليل شامل", use_container_width=True)
        if movie_name:
            movies_list = [movie_name]
    
    # تحليل شخصية (اختياري)
    analyze_character = st.checkbox("🧑‍🎤 تحليل شخصية معينة", value=False)
    character_name = None
    if analyze_character:
        character_name = st.text_input("اسم الشخصية:", placeholder="مثال: The Joker")
    
    # بدء التحليل
    if analyze_btn and movies_list:
        if not all(movies_list):
            st.error("الرجاء إدخال جميع الأسماء.")
            st.stop()
        
        # شريط تقدم متعدد المراحل
        progress_bar = st.progress(0, text="جاري تجهيز البيانات...")
        
        # مرحلة 1: جلب بيانات TMDB (إذا توفر المفتاح)
        tmdb_datas = []
        if 'tmdb_key' in st.session_state:
            for i, movie in enumerate(movies_list):
                progress_bar.progress((i+1)/(len(movies_list)*2), text=f"جلب بيانات TMDB لـ {movie}...")
                if st.session_state['content_type'] == "فيلم":
                    tmdb_data = fetch_tmdb_data(st.session_state['tmdb_key'], movie)
                else:
                    tmdb_data = fetch_tv_data(st.session_state['tmdb_key'], movie)
                tmdb_datas.append(tmdb_data)
        
        # مرحلة 2: الاتصال بـ Gemini
        progress_bar.progress(0.5, text="جاري الاتصال بمحرك التحليل...")
        
        if st.session_state.get('comparison_mode', False):
            # مقارنة متعددة
            other_movies = movies_list[1:]
            result = analyze_movie(
                st.session_state['gemini_key'],
                movies_list[0],
                content_type=st.session_state['content_type'],
                comparison_mode=True,
                other_movies=other_movies
            )
            
            if result and 'movies' in result:
                progress_bar.progress(1.0, text="اكتمل!")
                time.sleep(0.5)
                progress_bar.empty()
                
                # حفظ التقارير في قاعدة البيانات
                for report in result['movies']:
                    save_report_to_db(report)
                
                # عرض المقارنة
                st.markdown("---")
                st.header("📊 نتيجة المقارنة")
                
                movies_reports = result['movies']
                comparison = result.get('comparison', {})
                
                # بيانات للرسم البياني
                names = [r.info.arabic_title for r in movies_reports]
                scores = [r.recommendation.score for r in movies_reports]
                
                # رسم بياني أعمدة
                fig = px.bar(x=names, y=scores, title="تقييمات الأفلام", labels={'x':'الفيلم', 'y':'التقييم'}, range_y=[0,10])
                st.plotly_chart(fig, use_container_width=True)
                
                # جدول تفصيلي
                comparison_data = {
                    'العنصر': ['القصة', 'الأداء التمثيلي', 'الإخراج', 'الموسيقى', 'الرمزية']
                }
                for report in movies_reports:
                    comparison_data[report.info.arabic_title] = [
                        report.analysis.screenplay[:150] + '...',
                        report.analysis.acting[:150] + '...',
                        report.analysis.visuals[:150] + '...',
                        report.analysis.music[:150] + '...',
                        report.analysis.symbolism[:150] + '...'
                    ]
                
                df_comp = pd.DataFrame(comparison_data)
                st.dataframe(df_comp, use_container_width=True)
                
                # الحكم النهائي
                st.success(f"**الفائز الإجمالي:** {comparison.get('overall_winner', '')}")
                st.info(comparison.get('verdict', ''))
                
                # تحليل الفروقات
                st.subheader("🔍 تحليل الفروقات")
                diff_text = f"**أفضل قصة:** {comparison.get('better_plot', '')}\n\n"
                diff_text += f"**أفضل أداء:** {comparison.get('better_acting', '')}\n\n"
                diff_text += f"**أفضل إخراج:** {comparison.get('better_visuals', '')}\n\n"
                diff_text += f"**أفضل موسيقى:** {comparison.get('better_music', '')}"
                st.markdown(diff_text)
        
        else:
            # وضع عادي
            report = analyze_movie(
                st.session_state['gemini_key'],
                movies_list[0],
                content_type=st.session_state['content_type']
            )
            
            if report:
                progress_bar.progress(0.75, text="معالجة النتائج...")
                
                # حفظ في قاعدة البيانات
                save_report_to_db(report)
                
                tmdb_data = tmdb_datas[0] if tmdb_datas else None
                
                # عرض المقطع الدعائي إن وجد
                if tmdb_data and tmdb_data.get('trailer_key'):
                    st.video(f"https://www.youtube.com/watch?v={tmdb_data['trailer_key']}")
                
                # --- رأس الصفحة ---
                st.markdown("---")
                col_img, col_meta = st.columns([1, 3])
                
                with col_img:
                    if tmdb_data and tmdb_data.get('poster'):
                        st.image(tmdb_data['poster'], width=200)
                    else:
                        st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
                
                with col_meta:
                    # استخدام خلفية إذا وجدت
                    if tmdb_data and tmdb_data.get('backdrop'):
                        st.markdown(f"<div style='background-image: url({tmdb_data['backdrop']}); background-size: cover; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("العنوان", report.info.arabic_title)
                    c2.metric("السنة", report.info.year)
                    c3.metric("المخرج", report.info.director)
                    c4.metric("التقييم", f"{report.recommendation.score}/10")
                    
                    st.write("**التصنيف:** " + ", ".join([f"`{g}`" for g in report.info.genre]))
                    
                    if tmdb_data:
                        st.write(f"**تقييم TMDB:** {tmdb_data.get('rating', 'N/A')}/10")
                        if tmdb_data.get('cast'):
                            st.write("**أبرز الممثلين:** " + ", ".join(tmdb_data['cast']))
                    
                    if tmdb_data and tmdb_data.get('backdrop'):
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # تحليل شخصية إذا طلب
                if character_name:
                    with st.spinner(f"جاري تحليل شخصية {character_name}..."):
                        # يمكن إضافة طلب منفصل لتحليل الشخصية
                        st.info("هذه الميزة قيد التطوير، سيتم إضافتها قريباً.")
                
                # --- التبويبات ---
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 التحليل الفني", "⚖️ الحكم والمميزات", "🧠 العمق والرسائل", "🔗 توصيات ذكية", "📤 مشاركة وتصدير"])
                
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
                    st.write("**🤔 أفلام مشابهة (حسب التحليل):** " + ", ".join(report.recommendation.similar_movies))
                
                with tab3:
                    st.header("ما وراء الصورة")
                    st.markdown(f"> {report.analysis.symbolism}")
                    st.progress(report.recommendation.score / 10, text="جودة العمل الفني")
                
                with tab4:
                    st.header("توصيات ذكية")
                    recommendations = get_smart_recommendations(report, tmdb_data)
                    if recommendations:
                        for rec in recommendations:
                            if rec['source'] == "TMDB":
                                st.write(f"- 🎬 **{rec['title']}** (من TMDB)")
                            else:
                                st.write(f"- 🎥 **{rec['title']}** ({rec.get('director', '')}) – تقييم: {rec.get('score', 'N/A')}/10")
                    else:
                        st.info("قم بتحليل المزيد من الأفلام للحصول على توصيات مخصصة.")
                
                with tab5:
                    st.header("مشاركة وتصدير")
                    md_report = generate_markdown_report(report, tmdb_data)
                    st.download_button("📥 تحميل التقرير (Markdown)", data=md_report, file_name=f"{report.info.original_title}.md", mime="text/markdown")
                    
                    # نسخ الرابط (محاكاة)
                    if st.button("📋 نسخ رابط المشاركة"):
                        st.info("تم نسخ الرابط (محاكاة)، يمكنك مشاركته مع أصدقائك.")
                
                progress_bar.progress(1.0, text="اكتمل!")
                time.sleep(0.5)
                progress_bar.empty()

if __name__ == "__main__":
    main()
