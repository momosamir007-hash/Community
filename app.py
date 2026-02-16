
import streamlit as st
import requests
import json
import time
import sqlite3
import pandas as pd
import plotly.express as px
import re
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any, Union

# ==========================================
# 1. إعدادات الصفحة والتصميم
# ==========================================
st.set_page_config(
    page_title="CineMate Pro - الناقد السينمائي",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_theme(theme):
    """تطبيق السمات (Themes) وتخصيص CSS"""
    if theme == "داكن":
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
        border_color = "#3b3b3b"
    else:
        bg_color = "#ffffff"
        text_color = "#31333F"
        card_bg = "#f0f2f6"
        border_color = "#cccccc"
    
    st.markdown(f"""
    <style>
        .main {{direction: rtl; text-align: right; background-color: {bg_color}; color: {text_color};}}
        .stTextInput > div > div > input {{text-align: right;}}
        .stTextArea > div > div > textarea {{text-align: right;}}
        h1, h2, h3, h4, p, span, div {{font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}}
        .metric-card {{
            background-color: {card_bg}; 
            padding: 15px; 
            border-radius: 10px; 
            border: 1px solid {border_color}; 
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .stTabs [data-baseweb="tab-list"] {{ justify-content: flex-end; }}
        .stTabs [data-baseweb="tab"] {{ font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. نماذج البيانات (Pydantic Models)
# ==========================================
class MovieInfo(BaseModel):
    arabic_title: str = Field(..., description="عنوان الفيلم بالعربية")
    original_title: str = Field(..., description="العنوان الأصلي")
    year: Union[int, str] = Field(..., description="سنة الإصدار")
    director: str = Field(..., description="اسم المخرج")
    duration: str = Field(..., description="المدة")
    genre: List[str] = Field(..., description="قائمة الأنواع بالعربية")
    type: str = Field("فيلم", description="فيلم أو مسلسل")

class TechnicalAnalysis(BaseModel):
    screenplay: str = Field(..., description="تحليل السيناريو والحبكة")
    acting: str = Field(..., description="تحليل الأداء التمثيلي")
    visuals: str = Field(..., description="الإخراج والبصريات")
    music: str = Field(..., description="الموسيقى والصوت")
    symbolism: str = Field(..., description="الرمزية والرسائل الخفية")

class Recommendation(BaseModel):
    score: float = Field(..., description="التقييم من 10")
    pros: List[str] = Field(..., description="أبرز 3 إيجابيات")
    cons: List[str] = Field(..., description="أبرز 3 سلبيات")
    similar_movies: List[str] = Field(..., description="3 أعمال مشابهة")
    streaming_on: List[str] = Field(..., description="منصات المشاهدة المقترحة")
    final_verdict: str = Field(..., description="الحكم النهائي المختصر")

class FullMovieReport(BaseModel):
    info: MovieInfo
    analysis: TechnicalAnalysis
    recommendation: Recommendation

class ComparisonData(BaseModel):
    better_plot: str
    better_acting: str
    better_visuals: str
    better_music: str
    overall_winner: str
    verdict: str

class ComparisonResult(BaseModel):
    movies: List[FullMovieReport]
    comparison: ComparisonData

# ==========================================
# 3. قاعدة البيانات (SQLite)
# ==========================================
DB_FILE = 'cinemate_v2.db'

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
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
    except sqlite3.Error as e:
        st.error(f"خطأ في قاعدة البيانات: {e}")
    finally:
        if 'conn' in locals(): conn.close()

def save_report_to_db(report: FullMovieReport):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # تحويل السنة إلى رقم إذا كانت نصاً، أو 0 إذا فشل التحويل
        try:
            year_val = int(str(report.info.year).strip())
        except:
            year_val = 0

        c.execute('''INSERT INTO reports (title, arabic_title, director, genres, score, year, type)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (report.info.original_title,
                   report.info.arabic_title,
                   report.info.director,
                   json.dumps(report.info.genre, ensure_ascii=False),
                   report.recommendation.score,
                   year_val,
                   report.info.type))
        conn.commit()
    except sqlite3.Error as e:
        st.warning(f"لم يتم حفظ التقرير في السجل المحلي: {e}")
    finally:
        if 'conn' in locals(): conn.close()

def get_reports_from_db(limit=50):
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))
        conn.close()
        
        if not df.empty:
            # معالجة آمنة لتحويل JSON
            def safe_json_loads(x):
                try:
                    return json.loads(x) if isinstance(x, str) else []
                except:
                    return []
            
            df['genres'] = df['genres'].apply(safe_json_loads)
        return df
    except Exception as e:
        st.error(f"خطأ أثناء جلب السجل: {e}")
        return pd.DataFrame()

init_db()

# ==========================================
# 4. دوال TMDB API
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tmdb_data(api_key: str, query: str, is_tv: bool = False):
    """جلب بيانات من TMDB (فيلم أو مسلسل)"""
    if not api_key:
        return None
    
    base_url = "https://api.themoviedb.org/3"
    endpoint = "tv" if is_tv else "movie"
    search_url = f"{base_url}/search/{endpoint}"
    
    try:
        # 1. البحث
        params = {"api_key": api_key, "query": query, "language": "ar-SA"}
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('results'):
            return None
            
        item = data['results'][0]
        item_id = item['id']
        
        # 2. التفاصيل (Credits & Videos)
        details_params = {"api_key": api_key, "append_to_response": "credits,videos,recommendations"}
        details_url = f"{base_url}/{endpoint}/{item_id}"
        
        details_resp = requests.get(details_url, params=details_params, timeout=10)
        details_resp.raise_for_status()
        details = details_resp.json()
        
        # استخراج البيانات
        cast = [p['name'] for p in details.get('credits', {}).get('cast', [])[:5]]
        
        director = "غير معروف"
        if not is_tv:
            crew = details.get('credits', {}).get('crew', [])
            director = next((c['name'] for c in crew if c['job'] == 'Director'), "غير معروف")
        else:
            created_by = details.get('created_by', [])
            if created_by:
                director = created_by[0]['name']

        trailer_key = None
        for vid in details.get('videos', {}).get('results', []):
            if vid['site'] == 'YouTube' and vid['type'] == 'Trailer':
                trailer_key = vid['key']
                break
        
        similar = [s['title'] if not is_tv else s['name'] for s in details.get('recommendations', {}).get('results', [])[:3]]

        return {
            'poster': f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None,
            'backdrop': f"https://image.tmdb.org/t/p/w1280{item.get('backdrop_path')}" if item.get('backdrop_path') else None,
            'rating': item.get('vote_average', 0),
            'overview': item.get('overview', ''),
            'cast': cast,
            'director': director,
            'trailer_key': trailer_key,
            'similar_tmdb': similar,
            'year': (item.get('release_date') or item.get('first_air_date') or "N/A")[:4]
        }
        
    except requests.exceptions.RequestException as e:
        # لا نوقف التطبيق إذا فشل TMDB، فقط نسجل تحذيراً
        print(f"TMDB Error: {e}") 
        return None

# ==========================================
# 5. محرك التحليل (Cerebras)
# ==========================================
def extract_json_from_text(text: str) -> Optional[dict]:
    """استخراج كائن JSON صالح من نص قد يحتوي على كلام إضافي"""
    try:
        # المحاولة الأولى: تحليل النص مباشرة
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # المحاولة الثانية: البحث عن نمط JSON بين أقواس {}
    # نبحث عن أول { وآخر }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
            
    return None

def analyze_media(api_key: str, queries: List[str], content_type: str = "فيلم", comparison: bool = False) -> Union[FullMovieReport, ComparisonResult, None]:
    """التواصل مع Cerebras API للتحليل"""
    
    # تحديد النموذج (Llama 3.1 70B هو الشائع حالياً على Cerebras)
    MODEL_NAME = "llama3.1-70b" 
    API_URL = "https://api.cerebras.ai/v1/chat/completions"
    
    media_str = " و ".join(queries)
    
    # بناء التعليمات (Prompt)
    if comparison:
        schema = json.dumps(ComparisonResult.model_json_schema(), indent=2, ensure_ascii=False)
        system_prompt = f"""
        You are a legendary Arab Film Critic. Compare these {content_type}s: {media_str}.
        Output STRICT JSON matching this schema:
        {schema}
        Language: Arabic. Do not add markdown backticks.
        """
    else:
        schema = json.dumps(FullMovieReport.model_json_schema(), indent=2, ensure_ascii=False)
        system_prompt = f"""
        You are a legendary Arab Film Critic. Analyze the {content_type}: "{media_str}".
        Output STRICT JSON matching this schema:
        {schema}
        Language: Arabic. Do not add markdown backticks.
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze: {media_str}"}
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        parsed_data = extract_json_from_text(content)
        if not parsed_data:
            raise ValueError("فشل في استخراج JSON صالح من الرد.")

        # التحقق من صحة البيانات باستخدام Pydantic
        if comparison:
            return ComparisonResult(**parsed_data)
        else:
            return FullMovieReport(**parsed_data)

    except requests.exceptions.HTTPError as http_err:
        st.error(f"خطأ في الاتصال بـ API: {http_err.response.status_code} - {http_err.response.text}")
    except ValidationError as val_err:
        st.error(f"خطأ في هيكلية البيانات المستلمة: {val_err}")
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {str(e)}")
    
    return None

# ==========================================
# 6. الواجهة الرئيسية (Main)
# ==========================================
def main():
    # --- الشريط الجانبي ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=80)
        st.title("الإعدادات")
        
        # التعامل مع المفاتيح
        cerebras_key = st.session_state.get('cerebras_key', '')
        tmdb_key = st.session_state.get('tmdb_key', '')
        
        # محاولة جلب المفاتيح من secrets إذا لم تكن موجودة
        if not cerebras_key and "cerebras_key" in st.secrets:
            cerebras_key = st.secrets["cerebras_key"]
        
        if not tmdb_key and "tmdb_key" in st.secrets:
            tmdb_key = st.secrets["tmdb_key"]

        # حقول الإدخال
        new_c_key = st.text_input("Cerebras API Key", value=cerebras_key, type="password")
        new_t_key = st.text_input("TMDB API Key (اختياري)", value=tmdb_key, type="password")
        
        if new_c_key: st.session_state['cerebras_key'] = new_c_key
        if new_t_key: st.session_state['tmdb_key'] = new_t_key
        
        st.divider()
        
        theme = st.selectbox("المظهر", ["فاتح", "داكن"])
        apply_theme(theme)
        
        content_type = st.radio("نوع المحتوى", ["فيلم", "مسلسل"], horizontal=True)
        comparison_mode = st.checkbox("وضع المقارنة (أكثر من عمل)")
        
        num_movies = 1
        if comparison_mode:
            num_movies = st.number_input("عدد الأعمال", min_value=2, max_value=4, value=2)

        st.divider()
        st.subheader("سجل البحث")
        history_df = get_reports_from_db(5)
        if not history_df.empty:
            for _, row in history_df.iterrows():
                st.caption(f"🎬 {row['arabic_title']} ({row['score']}/10)")
        else:
            st.caption("لا يوجد سجلات بعد.")

    # --- المحتوى الرئيسي ---
    st.title("🎬 CineMate Pro")
    st.markdown("#### منصة النقد السينمائي المدعومة بالذكاء الاصطناعي")
    
    if not st.session_state.get('cerebras_key'):
        st.warning("⚠️ يرجى إدخال مفتاح Cerebras API في القائمة الجانبية للبدء.")
        st.stop()

    # حقول الإدخال
    movies_list = []
    cols = st.columns(num_movies)
    for i, col in enumerate(cols):
        with col:
            placeholder = "مثال: The Godfather" if i == 0 else "مثال: Goodfellas"
            val = st.text_input(f"العمل رقم {i+1}", key=f"movie_in_{i}", placeholder=placeholder)
            if val: movies_list.append(val)

    if st.button("🚀 بدء التحليل الشامل", use_container_width=True):
        if len(movies_list) < num_movies:
            st.error(f"يرجى إدخال أسماء {num_movies} أعمال.")
        else:
            main_placeholder = st.empty()
            with main_placeholder.container():
                st.info("جاري الاتصال بقواعد البيانات وتحليل المحتوى... يرجى الانتظار.")
                progress = st.progress(0)
                
                # 1. جلب بيانات TMDB (توازي وهمي عبر التكرار السريع)
                tmdb_results = []
                for idx, movie in enumerate(movies_list):
                    progress.progress((idx + 1) * 10, text=f"جلب بيانات Metadata لـ: {movie}")
                    t_data = fetch_tmdb_data(st.session_state.get('tmdb_key'), movie, is_tv=(content_type=="مسلسل"))
                    tmdb_results.append(t_data)
                
                # 2. تحليل الذكاء الاصطناعي
                progress.progress(50, text="جاري التحليل النقدي العميق (Cerebras AI)...")
                analysis_result = analyze_media(
                    st.session_state['cerebras_key'], 
                    movies_list, 
                    content_type=content_type, 
                    comparison=comparison_mode
                )
                
                progress.progress(100, text="اكتمل!")
                time.sleep(0.5)
                progress.empty()
                main_placeholder.empty()

                if analysis_result:
                    # تحويل النتيجة الفردية إلى قائمة لتوحيد العرض
                    reports = []
                    comp_data = None
                    
                    if comparison_mode and isinstance(analysis_result, ComparisonResult):
                        reports = analysis_result.movies
                        comp_data = analysis_result.comparison
                    elif isinstance(analysis_result, FullMovieReport):
                        reports = [analysis_result]

                    # حفظ النتائج
                    for r in reports:
                        save_report_to_db(r)

                    # --- عرض النتائج ---
                    
                    # إذا كانت مقارنة، عرض قسم المقارنة أولاً
                    if comparison_mode and comp_data:
                        st.header("⚖️ نتيجة المقارنة")
                        w_col1, w_col2 = st.columns([1, 2])
                        with w_col1:
                            st.metric("🏆 الفائز الإجمالي", comp_data.overall_winner)
                        with w_col2:
                            st.info(f"**الحكم:** {comp_data.verdict}")
                        
                        # جدول المقارنة
                        comp_df = pd.DataFrame({
                            "المعيار": ["القصة", "التمثيل", "البصريات", "الموسيقى"],
                            "الأفضل": [comp_data.better_plot, comp_data.better_acting, comp_data.better_visuals, comp_data.better_music]
                        })
                        st.table(comp_df)
                        
                        # رسم بياني للتقييمات
                        scores = {r.info.arabic_title: r.recommendation.score for r in reports}
                        fig = px.bar(
                            x=list(scores.keys()), 
                            y=list(scores.values()), 
                            labels={'x':'العمل', 'y':'التقييم'},
                            title="مقارنة التقييمات",
                            color=list(scores.values()),
                            color_continuous_scale='Viridis',
                            range_y=[0, 10]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.divider()

                    # عرض تفاصيل كل فيلم
                    for i, report in enumerate(reports):
                        t_data = tmdb_results[i] if i < len(tmdb_results) else None
                        
                        with st.container():
                            # رأس البطاقة
                            col_img, col_txt = st.columns([1, 3])
                            with col_img:
                                if t_data and t_data.get('poster'):
                                    st.image(t_data['poster'], use_container_width=True)
                                else:
                                    st.markdown("📷 صورة غير متوفرة")
                            
                            with col_txt:
                                st.subheader(f"{report.info.arabic_title} ({report.info.year})")
                                st.caption(f"{report.info.original_title} | {report.info.director}")
                                
                                m1, m2, m3 = st.columns(3)
                                m1.metric("التقييم النقدي", f"{report.recommendation.score}/10")
                                m2.metric("النوع", ", ".join(report.info.genre[:2]))
                                if t_data:
                                    m3.metric("تقييم الجمهور (TMDB)", f"{t_data.get('rating')}/10")
                                
                                if t_data and t_data.get('trailer_key'):
                                    with st.expander("🎥 مشاهدة الإعلان التشويقي"):
                                        st.video(f"https://www.youtube.com/watch?v={t_data['trailer_key']}")

                            # تبويبات التفاصيل
                            tab1, tab2, tab3 = st.tabs(["التحليل الفني", "الإيجابيات والسلبيات", "التوصيات"])
                            
                            with tab1:
                                st.markdown(f"**السيناريو:** {report.analysis.screenplay}")
                                st.markdown(f"**التمثيل:** {report.analysis.acting}")
                                st.markdown(f"**البصريات:** {report.analysis.visuals}")
                                st.markdown(f"**الموسيقى:** {report.analysis.music}")
                                st.markdown(f"--- \n **💡 الرمزية:** {report.analysis.symbolism}")

                            with tab2:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.success("✅ **نقاط القوة**")
                                    for p in report.recommendation.pros: st.write(f"- {p}")
                                with c2:
                                    st.error("❌ **نقاط الضعف**")
                                    for c in report.recommendation.cons: st.write(f"- {c}")
                                st.markdown(f"**📝 الحكم النهائي:** {report.recommendation.final_verdict}")

                            with tab3:
                                st.write("**📺 منصات مقترحة:** " + "، ".join(report.recommendation.streaming_on))
                                st.write("**🔗 أعمال مشابهة (AI):** " + "، ".join(report.recommendation.similar_movies))
                                if t_data and t_data.get('similar_tmdb'):
                                    st.write("**🔗 أعمال مشابهة (TMDB):** " + "، ".join(t_data['similar_tmdb']))

                        st.divider()

if __name__ == "__main__":
    main()
