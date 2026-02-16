import streamlit as st
import requests
import json
import time
from pydantic import BaseModel, Field
from typing import List, Optional
from collections import Counter

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
    .tmdb-card {background-color: #0e1a2b; color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    .comparison-table {background-color: #f9f9f9; border-radius: 10px; padding: 10px;}
    .cast-card {display: inline-block; text-align: center; margin: 5px; width: 100px;}
    .cast-card img {border-radius: 50%; width: 80px; height: 80px; object-fit: cover;}
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
# 3. دوال مساعدة (TMDB Integration متقدم)
# ==========================================
def fetch_tmdb_data(api_key: str, movie_name: str):
    """جلب بيانات غنية من TMDB: تفاصيل، طاقم، صور، توصيات، فيديوهات"""
    if not api_key:
        return None
    try:
        # بحث متعدد اللغات (إنجليزي + عربي)
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": api_key,
            "query": movie_name,
            "language": "ar-SA",  # نحاول العربية أولاً
            "include_adult": False
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data['results']:
            # جرب بالإنجليزية إذا لم تكن هناك نتائج عربية
            params["language"] = "en-US"
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data['results']:
                return None
        
        movie = data['results'][0]
        movie_id = movie['id']
        
        # جلب التفاصيل الكاملة
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        details_params = {
            "api_key": api_key,
            "language": "ar-SA",
            "append_to_response": "credits,videos,recommendations,release_dates"
        }
        details_response = requests.get(details_url, params=details_params)
        details_response.raise_for_status()
        details = details_response.json()
        
        # معالجة البيانات
        poster = f"https://image.tmdb.org/t/p/w500{details['poster_path']}" if details.get('poster_path') else None
        backdrop = f"https://image.tmdb.org/t/p/original{details['backdrop_path']}" if details.get('backdrop_path') else None
        
        # طاقم التمثيل (أول 10 مع صور)
        cast = []
        for actor in details.get('credits', {}).get('cast', [])[:10]:
            cast.append({
                'name': actor['name'],
                'character': actor['character'],
                'profile': f"https://image.tmdb.org/t/p/w185{actor['profile_path']}" if actor.get('profile_path') else None,
                'order': actor['order']
            })
        
        # المخرج والكتاب
        director = None
        writers = []
        for crew in details.get('credits', {}).get('crew', []):
            if crew['job'] == 'Director':
                director = crew['name']
            elif crew['job'] in ['Writer', 'Screenplay', 'Author']:
                writers.append(crew['name'])
        
        # فيديوهات (trailer)
        videos = []
        for video in details.get('videos', {}).get('results', []):
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                videos.append({
                    'key': video['key'],
                    'name': video['name']
                })
        
        # توصيات من TMDB
        recommendations = []
        for rec in details.get('recommendations', {}).get('results', [])[:5]:
            recommendations.append({
                'title': rec['title'],
                'poster': f"https://image.tmdb.org/t/p/w200{rec['poster_path']}" if rec.get('poster_path') else None,
                'year': rec.get('release_date', '')[:4] if rec.get('release_date') else None,
                'id': rec['id']
            })
        
        # تصنيف المحتوى (PG-13, R, إلخ) حسب البلد
        certification = None
        for release in details.get('release_dates', {}).get('results', []):
            if release['iso_3166_1'] == 'US':  # نأخذ التصنيف الأمريكي كمرجع
                for rel in release['release_dates']:
                    if rel.get('certification'):
                        certification = rel['certification']
                        break
                if certification:
                    break
        
        return {
            'id': movie_id,
            'poster': poster,
            'backdrop': backdrop,
            'rating': details.get('vote_average'),
            'votes': details.get('vote_count'),
            'overview': details.get('overview'),
            'tagline': details.get('tagline'),
            'budget': details.get('budget'),
            'revenue': details.get('revenue'),
            'runtime': details.get('runtime'),
            'original_language': details.get('original_language'),
            'production_countries': [c['name'] for c in details.get('production_countries', [])],
            'genres': [g['name'] for g in details.get('genres', [])],
            'cast': cast,
            'director': director,
            'writers': writers,
            'videos': videos,
            'recommendations': recommendations,
            'certification': certification,
            'homepage': details.get('homepage')
        }
    except Exception as e:
        st.warning(f"تعذر جلب بيانات TMDB: {e}")
        return None

def format_currency(amount):
    """تنسيق الأرقام كعملة (دولار)"""
    if not amount or amount == 0:
        return "غير متوفر"
    return f"${amount:,.0f}"

# ==========================================
# 4. محرك التحليل (Cerebras)
# ==========================================
def analyze_movie(api_key: str, movie_name: str, comparison_mode: bool = False, second_movie: str = None) -> Optional[FullMovieReport]:
    """
    يتصل بـ Cerebras API ويحلل الفيلم
    إذا كان comparison_mode = True، يطلب تحليل فيلمين معاً
    """
    API_URL = "https://api.cerebras.ai/v1/chat/completions"
    MODEL = "llama-3.3-70b"
    
    schema_json = json.dumps(FullMovieReport.model_json_schema(), indent=2)
    
    if comparison_mode and second_movie:
        system_content = f"""
        You are an elite Arab Film Critic. Compare the two movies: '{movie_name}' and '{second_movie}'.
        First, analyze each movie separately according to the schema, then provide a comparison table.
        Output MUST be a JSON object with keys: 'movie1', 'movie2', 'comparison'.
        Each movie should follow the FullMovieReport schema, and 'comparison' should be a dict with keys: 'better_plot', 'better_acting', 'better_visuals', 'better_music', 'overall_winner', 'verdict'.
        Language: Arabic.
        """
        user_content = f"Compare {movie_name} and {second_movie} in depth."
    else:
        system_content = f"""
        You are an elite Arab Film Critic (like Youssef Chahine mixed with Roger Ebert).
        Analyze the requested movie/series deeply.
        Language: High-quality Arabic (Fusha).
        You MUST output strict JSON following this schema:
        {schema_json}
        """
        user_content = f"Analyze: {movie_name}"
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
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
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']
        parsed_data = json.loads(content)
        
        if comparison_mode and second_movie:
            return parsed_data  # سيكون قاموساً بالمقارنة
        else:
            return FullMovieReport(**parsed_data)
    except Exception as e:
        st.error(f"حدث خطأ أثناء الاتصال: {str(e)}")
        if 'response' in locals():
            st.code(response.text)
        return None

# ==========================================
# 5. دوال التوصيات الذكية
# ==========================================
def update_recommendations(new_report):
    """تحديث سجل التوصيات بناءً على التحليل الجديد"""
    if 'reports_history' not in st.session_state:
        st.session_state['reports_history'] = []
    
    # نضيف التقرير الجديد مع بعض المعلومات المبسطة
    st.session_state['reports_history'].append({
        'title': new_report.info.original_title,
        'arabic_title': new_report.info.arabic_title,
        'director': new_report.info.director,
        'genres': new_report.info.genre,
        'score': new_report.recommendation.score
    })

def get_smart_recommendations(current_report, top_n=3):
    """توليد توصيات بناءً على تشابه الأنواع والمخرج"""
    if 'reports_history' not in st.session_state or len(st.session_state['reports_history']) < 2:
        return []
    
    history = st.session_state['reports_history']
    current_genres = set(current_report.info.genre)
    current_director = current_report.info.director
    
    scores = []
    for idx, item in enumerate(history):
        if item['title'] == current_report.info.original_title:
            continue  # نتخطى الفيلم الحالي
        
        # حساب درجة التشابه
        genre_similarity = len(current_genres & set(item['genres'])) / max(len(current_genres), len(item['genres']))
        director_match = 1 if current_director == item['director'] else 0
        total_score = genre_similarity * 0.7 + director_match * 0.3
        
        scores.append((total_score, item))
    
    # ترتيب تنازلي وأخذ الأعلى
    scores.sort(reverse=True, key=lambda x: x[0])
    return [item for score, item in scores[:top_n]]

# ==========================================
# 6. واجهة التطبيق
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=100)
    st.title("إعدادات المحرك")
    
    # مفتاح Cerebras
    cerebras_key = st.text_input("مفتاح Cerebras API", type="password", help="يبدأ بـ csk-")
    if cerebras_key:
        st.session_state['cerebras_key'] = cerebras_key
    
    # مفتاح TMDB (اختياري)
    tmdb_key = st.text_input("مفتاح TMDB API (اختياري)", type="password", help="لجلب بيانات إضافية غنية")
    if tmdb_key:
        st.session_state['tmdb_key'] = tmdb_key
    
    # وضع المقارنة
    st.session_state['comparison_mode'] = st.checkbox("🔁 تفعيل وضع المقارنة", value=False)
    
    st.info("💡 يستخدم Llama-3.3-70b عبر Cerebras.")
    st.markdown("---")
    st.write("Designed by: **AI Architect**")

# التحقق من المفتاح الأساسي
if 'cerebras_key' not in st.session_state:
    st.warning("⚠️ يرجى إدخال مفتاح Cerebras API في القائمة الجانبية للبدء.")
    st.stop()

# الواجهة الرئيسية
st.title("🎬 CineMate Pro")
st.subheader("منصة التحليل السينمائي المتقدمة")

# حقول الإدخال حسب الوضع
if st.session_state.get('comparison_mode', False):
    col1, col2 = st.columns(2)
    with col1:
        movie1 = st.text_input("الفيلم الأول:", placeholder="مثال: The Godfather")
    with col2:
        movie2 = st.text_input("الفيلم الثاني:", placeholder="مثال: Scarface")
    analyze_btn = st.button("🔍 تحليل مقارن شامل", use_container_width=True)
    movie_name = movie1
    second_movie = movie2
else:
    movie_name = st.text_input("اسم الفيلم أو المسلسل:", placeholder="مثال: Interstellar")
    second_movie = None
    analyze_btn = st.button("🔍 تحليل شامل", use_container_width=True)

# منطق العرض
if analyze_btn:
    if st.session_state.get('comparison_mode', False):
        if not movie1 or not movie2:
            st.error("الرجاء إدخال اسمي الفيلمين للمقارنة.")
            st.stop()
        with st.spinner(f"جاري المقارنة بين '{movie1}' و '{movie2}'..."):
            comparison_result = analyze_movie(
                st.session_state['cerebras_key'], 
                movie1, 
                comparison_mode=True, 
                second_movie=movie2
            )
            if comparison_result:
                # عرض المقارنة
                st.markdown("---")
                st.header("📊 نتيجة المقارنة")
                
                movie1_data = comparison_result.get('movie1')
                movie2_data = comparison_result.get('movie2')
                comparison = comparison_result.get('comparison', {})
                
                if movie1_data and movie2_data:
                    # جدول المقارنة
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader(f"🎬 {movie1_data['info']['arabic_title']}")
                        st.metric("التقييم", f"{movie1_data['recommendation']['score']}/10")
                    with col_b:
                        st.subheader(f"🎬 {movie2_data['info']['arabic_title']}")
                        st.metric("التقييم", f"{movie2_data['recommendation']['score']}/10")
                    
                    # جدول تفصيلي
                    comparison_data = {
                        'العنصر': ['القصة', 'الأداء التمثيلي', 'الإخراج', 'الموسيقى', 'الرمزية'],
                        movie1_data['info']['arabic_title']: [
                            movie1_data['analysis']['screenplay'][:100] + '...',
                            movie1_data['analysis']['acting'][:100] + '...',
                            movie1_data['analysis']['visuals'][:100] + '...',
                            movie1_data['analysis']['music'][:100] + '...',
                            movie1_data['analysis']['symbolism'][:100] + '...'
                        ],
                        movie2_data['info']['arabic_title']: [
                            movie2_data['analysis']['screenplay'][:100] + '...',
                            movie2_data['analysis']['acting'][:100] + '...',
                            movie2_data['analysis']['visuals'][:100] + '...',
                            movie2_data['analysis']['music'][:100] + '...',
                            movie2_data['analysis']['symbolism'][:100] + '...'
                        ]
                    }
                    st.table(comparison_data)
                    
                    # الحكم النهائي
                    st.success(f"**الفائز الإجمالي:** {comparison.get('overall_winner', '')}")
                    st.info(comparison.get('verdict', ''))
    
    else:  # وضع عادي
        if not movie_name:
            st.error("الرجاء كتابة اسم الفيلم أولاً.")
            st.stop()
        
        with st.spinner(f"جاري استحضار النقد السينمائي لـ '{movie_name}'..."):
            report = analyze_movie(st.session_state['cerebras_key'], movie_name)
            
            if report:
                # تحديث سجل التوصيات
                update_recommendations(report)
                
                # جلب بيانات TMDB إذا توفر المفتاح
                tmdb_data = None
                if 'tmdb_key' in st.session_state:
                    tmdb_data = fetch_tmdb_data(st.session_state['tmdb_key'], movie_name)
                
                # --- رأس الصفحة مع خلفية إن وجدت ---
                if tmdb_data and tmdb_data['backdrop']:
                    st.image(tmdb_data['backdrop'], use_column_width=True)
                
                col_img, col_meta = st.columns([1, 3])
                
                with col_img:
                    if tmdb_data and tmdb_data['poster']:
                        st.image(tmdb_data['poster'], width=250)
                    else:
                        st.image("https://via.placeholder.com/250x375?text=No+Poster", width=250)
                    
                    if tmdb_data and tmdb_data['videos']:
                        st.markdown("**🎬 مشاهدة الإعلان:**")
                        for video in tmdb_data['videos'][:1]:
                            video_url = f"https://www.youtube.com/watch?v={video['key']}"
                            st.markdown(f"[{video['name']}]({video_url})")
                
                with col_meta:
                    st.markdown(f"# {report.info.arabic_title}")
                    if tmdb_data and tmdb_data['tagline']:
                        st.markdown(f"*{tmdb_data['tagline']}*")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("العنوان الأصلي", report.info.original_title)
                    c2.metric("السنة", report.info.year)
                    c3.metric("المخرج", report.info.director)
                    c4.metric("تقييم CineMate", f"{report.recommendation.score}/10")
                    
                    # صف ثاني من المقاييس من TMDB
                    if tmdb_data:
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("تقييم TMDB", f"{tmdb_data['rating']}/10" if tmdb_data['rating'] else "N/A")
                        rc2.metric("عدد التقييمات", tmdb_data['votes'] if tmdb_data['votes'] else "N/A")
                        rc3.metric("المدة", f"{tmdb_data['runtime']} دقيقة" if tmdb_data['runtime'] else "N/A")
                        rc4.metric("التصنيف", tmdb_data['certification'] if tmdb_data['certification'] else "N/A")
                    
                    st.write("**التصنيف:** " + ", ".join([f"`{g}`" for g in report.info.genre]))
                    
                    if tmdb_data:
                        if tmdb_data['production_countries']:
                            st.write("**دول الإنتاج:** " + ", ".join(tmdb_data['production_countries']))
                        if tmdb_data['budget'] and tmdb_data['revenue']:
                            st.write(f"**الميزانية:** {format_currency(tmdb_data['budget'])}  |  **الإيرادات:** {format_currency(tmdb_data['revenue'])}")
                        if tmdb_data['homepage']:
                            st.markdown(f"**[الموقع الرسمي]({tmdb_data['homepage']})**")
                
                # --- تبويب خاص ببيانات TMDB ---
                if tmdb_data:
                    with st.expander("📽️ بيانات إضافية من TMDB", expanded=False):
                        if tmdb_data['cast']:
                            st.subheader("طاقم التمثيل")
                            cast_cols = st.columns(5)
                            for i, actor in enumerate(tmdb_data['cast'][:10]):
                                with cast_cols[i % 5]:
                                    if actor['profile']:
                                        st.image(actor['profile'], width=100)
                                    else:
                                        st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
                                    st.markdown(f"**{actor['name']}**")
                                    st.caption(actor['character'])
                        
                        if tmdb_data['writers']:
                            st.subheader("كتاب السيناريو")
                            st.write(", ".join(tmdb_data['writers']))
                        
                        if tmdb_data['recommendations']:
                            st.subheader("🔗 توصيات من TMDB")
                            rec_cols = st.columns(5)
                            for i, rec in enumerate(tmdb_data['recommendations'][:5]):
                                with rec_cols[i]:
                                    if rec['poster']:
                                        st.image(rec['poster'], width=120)
                                    else:
                                        st.image("https://via.placeholder.com/120x180?text=No+Poster", width=120)
                                    st.markdown(f"**{rec['title']}** ({rec['year']})")
                
                # --- التبويبات الأساسية ---
                tab1, tab2, tab3, tab4 = st.tabs(["📝 التحليل الفني", "⚖️ الحكم والمميزات", "🧠 العمق والرسائل", "🔗 توصيات ذكية"])
                
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
                    st.header("توصيات ذكية بناءً على تحليلاتك السابقة")
                    recommendations = get_smart_recommendations(report)
                    if recommendations:
                        for rec in recommendations:
                            st.write(f"- **{rec['arabic_title']}** ({rec['director']}) – تقييم: {rec['score']}/10")
                    else:
                        st.info("قم بتحليل المزيد من الأفلام للحصول على توصيات مخصصة.")
