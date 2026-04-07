import streamlit as st
import joblib
import pandas as pd
import requests
import re
import os
import gc
import nltk
import random
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from thefuzz import process as fuzz_process
import nltk

st.set_page_config(page_title="Cinemalyze", page_icon="🎬", layout="wide")

# The POS Tagger fix for Python 3.13
nltk.download('averaged_perceptron_tagger_eng')

# The Tokenizer fix for newer NLTK versions
nltk.download('punkt_tab')

# Standard essentials for your sentiment logic
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw_latin1')
nltk.download('omw-1.4')

# --- SESSION STATE INIT ---
if 'target_id' not in st.session_state:
    st.session_state.target_id = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'page' not in st.session_state:
    # Standardized to match the radio/if-logic in your provided code
    st.session_state.page = "Main Analytics"

# --- MODEL LOADING ---
@st.cache_resource
def load_ai_brain():
    m_lr = joblib.load('sentiment_model.joblib')
    m_et = joblib.load('extra_tree_model.joblib')
    vec = joblib.load('tfidf_vectorizer.joblib')
    gc.collect()
    return m_lr, m_et, vec

try:
    model_lr, model_et, tfidf_vectorizer = load_ai_brain()
except Exception as e:
    st.error(f"Error loading AI models: {e}. Ensure .joblib files are in the directory.")

# --- CONFIG ---
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "f1d69efb3938a73c4aee8a756489171d")

# ── DESIGN SYSTEM ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

:root {
    --gold:    #F5C518;
    --red:     #E50914;
    --sky:     #87CEEB;
    --border:  rgba(245,197,24,0.18);
    --pos:     #87CEEB;
    --neg:     #E50914;
    /* These inherit from Streamlit's native theme — flip with Light/Dark automatically */
    --text:    var(--text-color);
    --muted:   color-mix(in srgb, var(--text-color) 60%, transparent);
    --card:    color-mix(in srgb, var(--text-color) 5%, transparent);
    --surface: color-mix(in srgb, var(--text-color) 3%, transparent);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    /* No hardcoded background-color — Streamlit controls this natively */
}

.main .block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    letter-spacing: 0.06em;
    line-height: 1;
    background: linear-gradient(135deg, #F5C518 0%, #ffffff 50%, #87CEEB 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 2rem;
}

[data-testid="stTextInput"] input {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,197,24,0.12) !important;
}
[data-testid="stTextInput"] label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: var(--gold) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.65rem 2.5rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(245,197,24,0.35) !important;
}

[data-testid="stButton"] > button:not([kind="primary"]) {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 0.6rem !important;
    width: 100% !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: var(--gold) !important;
    background: rgba(245,197,24,0.07) !important;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    transition: transform 0.2s, border-color 0.2s !important;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.03) !important;
    border-color: var(--gold) !important;
}

[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.5rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important; color: var(--gold) !important; letter-spacing: 0.05em !important; }

.movie-title-large {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    letter-spacing: 0.05em;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.movie-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
}
.genre-pill {
    display: inline-block;
    background: rgba(245,197,24,0.12);
    border: 1px solid rgba(245,197,24,0.3);
    color: var(--gold);
    border-radius: 100px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.rating-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(245,197,24,0.1);
    border: 1px solid var(--gold);
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: var(--gold);
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}
.overview-text {
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--muted);
    border-left: 3px solid var(--gold);
    padding-left: 1rem;
    margin: 1rem 0;
}
.crew-line {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.crew-line span { color: var(--text); font-weight: 500; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid var(--border) !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

[data-testid="stCaptionContainer"] p { font-size: 0.72rem !important; color: var(--muted) !important; text-align: center; }

.trend-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: rgba(245,197,24,0.15);
    line-height: 1;
    margin-bottom: -0.5rem;
}
.trend-rating {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--gold);
    margin-top: 0.2rem;
}

.footer {
    text-align: center;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ── MOVIE DATABASE ─────────────────────────────────────────────────────────────
ALL_MOVIES_DB = [
    {"title": "The Dark Knight",       "image_url": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg"},
    {"title": "Inception",             "image_url": "https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg"},
    {"title": "Interstellar",          "image_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"},
    {"title": "Parasite",              "image_url": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg"},
    {"title": "The Matrix",            "image_url": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"},
    {"title": "Avatar",                "image_url": "https://image.tmdb.org/t/p/w500/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg"},
    {"title": "Joker",                 "image_url": "https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg"},
    {"title": "Gladiator",             "image_url": "https://image.tmdb.org/t/p/w500/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg"},
    {"title": "Titanic",               "image_url": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg"},
    {"title": "Avengers: Endgame",     "image_url": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg"},
    {"title": "Pulp Fiction",          "image_url": "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg"},
   # {"title": "Forrest Gump",          "image_url": "https://image.tmdb.org/t/p/w500/arw2vcBveWOVZr6pxm9H6zW2vR2.jpg"},
    {"title": "The Lion King",         "image_url": "https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg"},
    {"title": "Spirited Away",         "image_url": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg"},
    {"title": "Into the Spider-Verse", "image_url": "https://image.tmdb.org/t/p/w500/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg"},
  #  {"title": "3 Idiots",              "image_url": "https://image.tmdb.org/t/p/w500/66A9Mqez2sFOGVaObYTOHqYT24k.jpg"},
    {"title": "Dune",                  "image_url": "https://image.tmdb.org/t/p/w500/d5NXSklXo0qyIYkgV94XAgMIckC.jpg"},
   # {"title": "Sultan",                "image_url": "https://image.tmdb.org/t/p/w500/dTbBWRGXpGl7jJXPbRMBG9urTzG.jpg"},
]

if 'random_recs' not in st.session_state:
    st.session_state.random_recs = random.sample(ALL_MOVIES_DB, 5)

# ── NLP HELPERS ───────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_nlp_processing(text):
    clean = re.sub(r'<.*?>', ' ', str(text))
    clean = re.sub(r'[^a-zA-Z]', ' ', clean).lower()
    tokens = word_tokenize(clean)
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

def get_top_adjectives(texts):
    all_words = []
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        tags = nltk.pos_tag(tokens)
        adjectives = [w for w, t in tags if t in ['JJ', 'JJR', 'JJS'] and w not in stop_words and len(w) > 2]
        all_words.extend(adjectives)
    return Counter(all_words)

# ── API FUNCTIONS ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fuzzy_search_multi(query):
    """
    Calls search/multi, runs Levenshtein fuzzy match against all results,
    returns (best_result_dict, matched_title, score).
    """
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = session.get(
            "https://api.themoviedb.org/3/search/multi",
            params={"api_key": TMDB_API_KEY, "query": query, "include_adult": False},
            headers=headers, timeout=10
        )
        results = res.json().get('results', [])
        # Keep only movie/tv results that have a title or name
        candidates = [r for r in results if r.get('media_type') in ('movie', 'tv') and (r.get('title') or r.get('name'))]
        if not candidates:
            return None, None, 0

        # Build a title→result map for fuzzy matching
        title_map = {(r.get('title') or r.get('name')): r for r in candidates}
        best_title, score = fuzz_process.extractOne(query, list(title_map.keys()))
        return title_map[best_title], best_title, score
    except Exception:
        return None, None, 0


@st.cache_data(ttl=3600)
def get_full_movie_intelligence(query=None, movie_id=None):
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if not movie_id:
            # Use fuzzy multi-search to resolve the ID
            best, best_title, score = fuzzy_search_multi(query)
            if not best:
                return "NOT_FOUND"
            movie_id = best['id']

        full_res = session.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "append_to_response": "videos,credits,reviews,recommendations"},
            headers=headers, timeout=15
        )
        return full_res.json()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_trending_movies_robust():
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = session.get("https://api.themoviedb.org/3/trending/movie/week",
                          params={"api_key": TMDB_API_KEY}, headers=headers, timeout=10)
        return res.json().get('results', [])
    except:
        return []

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem;'>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;
                    letter-spacing:0.1em;color:#F5C518;'>CINEMALYZE</div>
        <div style='font-family:"DM Mono",monospace;font-size:0.6rem;
                    letter-spacing:0.2em;color:#64748b;text-transform:uppercase;'>
            Semantic AI Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    page_options = ["Main Analytics", "Trending Now"]
    current_idx = 0 if st.session_state.page == "Main Analytics" else 1
    st.session_state.page = st.sidebar.radio("Navigate to:", page_options, index=current_idx)

    st.markdown("<hr style='border-color:rgba(245,197,24,0.15);margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.65rem;
                letter-spacing:0.1em;color:#64748b;line-height:1.8;'>
        <div style='color:#F5C518;margin-bottom:0.5rem;'>MODELS ACTIVE</div>
        ▸ Logistic Regression<br>
        ▸ Extra Trees Classifier<br>
        ▸ TF-IDF Vectorizer<br><br>
        <div style='color:#F5C518;margin-bottom:0.5rem;'>DATA SOURCE</div>
        ▸ TMDb Live API<br>
        ▸ Real user reviews
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MAIN ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Main Analytics":

    st.markdown("<p class='hero-title'>CINEMALYZE</p>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>⬡ Semantic AI &nbsp;·&nbsp; Live Sentiment Intelligence &nbsp;·&nbsp; TMDb Integration</p>", unsafe_allow_html=True)

    movie_input = st.text_input(
        "SEARCH MOVIE",
        value=st.session_state.search_query,
        placeholder="e.g. Titanic, Inception, Dangal…"
    )

    st.markdown("<div class='section-label'>⬡ All Time Classics</div>", unsafe_allow_html=True)

    cols = st.columns(5)
    for i, movie in enumerate(st.session_state.random_recs):
        with cols[i]:
            st.image(movie['image_url'], use_container_width=True)
            if st.button(movie['title'], key=f"rec_btn_{i}", use_container_width=True):
                st.session_state.search_query = movie['title']
                st.session_state.target_id = None
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("▶  Run AI Engine", type="primary")
    target_movie = st.session_state.search_query if st.session_state.search_query else movie_input
    should_run = run_clicked or st.session_state.target_id or st.session_state.search_query

    if should_run and target_movie:
        # ── FUZZY MATCH CHECK (only when not using a direct movie_id) ──────
        if not st.session_state.target_id:
            _, best_title, score = fuzzy_search_multi(target_movie)
            if best_title and score < 95:
                st.info(f"🎬 Did you mean: **{best_title}**? Showing results for that title.")

        with st.spinner("Fetching intelligence…"):
            data = get_full_movie_intelligence(query=target_movie, movie_id=st.session_state.target_id)

        # UI BUG FIX: Display error if movie is not found or API fails
        if data == "NOT_FOUND":
            st.error(f"🎬 Oops! We couldn't find any movie titled '{target_movie}'. Please check the spelling.")
        elif not data:
            st.error("📡 Connection error. Please check your network or API key settings.")
        else:
            # ── MOVIE PROFILE ─────────────────────────────────────────────
            st.markdown("<div class='section-label'>⬡ Film Profile</div>", unsafe_allow_html=True)

            director = next((p['name'] for p in data['credits']['crew'] if p['job'] == 'Director'), "N/A")
            cast_names = ', '.join([p['name'] for p in data['credits']['cast'][:5]])
            genres_html = ''.join([f"<span class='genre-pill'>{g['name']}</span>" for g in data['genres']])
            trailers = [v for v in data['videos']['results'] if v['type'] == 'Trailer' and v['site'] == 'YouTube']

            c1, c2 = st.columns([1, 2.8], gap="large")
            with c1:
                if data.get('poster_path'):
                    st.image(f"https://image.tmdb.org/t/p/w500{data['poster_path']}", use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class='movie-title-large'>{data['title']}</div>
                <div class='movie-meta'>{data['release_date'][:4]} &nbsp;·&nbsp; {data.get('runtime','—')} min &nbsp;·&nbsp; {data.get('original_language','').upper()}</div>
                <div style='margin-bottom:1rem;'>{genres_html}</div>
                <div class='rating-badge'>★ &nbsp;{data['vote_average']:.1f}
                    <span style='font-family:"DM Sans",sans-serif;font-size:0.75rem;color:var(--muted);margin-left:0.4rem;'>
                        /10 &nbsp;({data.get('vote_count','—')} votes)
                    </span>
                </div>
                <div class='overview-text'>{data['overview']}</div>
                <div class='crew-line'>🎬 Director &nbsp;<span>{director}</span></div>
                <div class='crew-line'>🎭 Cast &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>{cast_names}</span></div>
                """, unsafe_allow_html=True)

                if trailers:
                    with st.expander("▶  Watch Trailer"):
                        st.video(f"https://www.youtube.com/watch?v={trailers[0]['key']}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── SENTIMENT ENGINE ──────────────────────────────────────────
            reviews = data['reviews'].get('results', [])

            if not reviews:
                st.warning("No user reviews found on TMDb for this title.")
            else:
                st.markdown("<div class='section-label'>⬡ AI Sentiment Analysis</div>", unsafe_allow_html=True)

                table_data = []
                pos_lr, pos_et = 0, 0
                p_content, n_content = [], []

                for r in reviews[:10]:
                    clean = advanced_nlp_processing(r['content'])
                    vec = tfidf_vectorizer.transform([clean])
                    p_lr = model_lr.predict(vec)[0]
                    p_et = model_et.predict(vec)[0]
                    conf = f"{max(model_lr.predict_proba(vec)[0])*100:.1f}%"

                    if p_lr == 'positive':
                        pos_lr += 1
                        p_content.append(r['content'])
                    else:
                        n_content.append(r['content'])
                    if p_et == 'positive':
                        pos_et += 1

                    table_data.append({
                        "Author": r['author'],
                        "Log. Regression": p_lr.upper(),
                        "Extra Trees": p_et.upper(),
                        "Confidence": conf,
                        "Full Review": r['content']
                    })

                total = len(table_data)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Reviews Analysed", total)
                m2.metric("LR Positive", f"{(pos_lr/total)*100:.0f}%")
                m3.metric("ET Positive", f"{(pos_et/total)*100:.0f}%")
                m4.metric("Model Agreement", "High ✓" if abs(pos_lr - pos_et) <= 1 else "Varying")

                st.markdown("<br>", unsafe_allow_html=True)

                # Chart + word traits
                v1, v2 = st.columns([1.2, 1], gap="large")
                with v1:
                    fig = go.Figure(go.Pie(
                        labels=["Positive", "Negative"],
                        values=[pos_lr, total - pos_lr],
                        hole=0.55,
                        marker_colors=["#87CEEB", "#E50914"],
                        textinfo='label+percent',
                        textfont=dict(family="DM Mono", size=12, color="white"),
                        hovertemplate="%{label}: %{value} reviews<extra></extra>"
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        margin=dict(t=10, b=10, l=10, r=10),
                        annotations=[dict(
                            text=f"<b>{(pos_lr/total)*100:.0f}%</b><br>positive",
                            x=0.5, y=0.5, font_size=18,
                            font_color="#F5C518",
                            font_family="Bebas Neue",
                            showarrow=False
                        )]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with v2:
                    pos_words = [w for w, _ in get_top_adjectives(p_content).most_common(6)]
                    neg_words = [w for w, _ in get_top_adjectives(n_content).most_common(6)]
                    st.markdown(f"""
                    <div style='margin-bottom:1.5rem;margin-top:2rem;'>
                        <div style='font-family:"DM Mono",monospace;font-size:0.65rem;
                                    letter-spacing:0.15em;text-transform:uppercase;
                                    color:#87CEEB;margin-bottom:0.6rem;'>● Positive Traits</div>
                        <div style='font-size:0.9rem;color:var(--text);line-height:1.8;'>
                            {", ".join(pos_words) if pos_words else "—"}
                        </div>
                    </div>
                    <div>
                        <div style='font-family:"DM Mono",monospace;font-size:0.65rem;
                                    letter-spacing:0.15em;text-transform:uppercase;
                                    color:#E50914;margin-bottom:0.6rem;'>● Negative Traits</div>
                        <div style='font-size:0.9rem;color:var(--text);line-height:1.8;'>
                            {", ".join(neg_words) if neg_words else "—"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Review table
                st.markdown("<div class='section-label' style='margin-top:1.5rem;'>⬡ Review Breakdown</div>", unsafe_allow_html=True)
                df = pd.DataFrame(table_data)

                def style_sent(val):
                    if val == 'POSITIVE': return 'color: #87CEEB; font-weight: 600;'
                    if val == 'NEGATIVE': return 'color: #E50914; font-weight: 600;'
                    return ''

                st.dataframe(
                    df.style.map(style_sent, subset=['Log. Regression', 'Extra Trees']),
                    use_container_width=True,
                    hide_index=True
                )

            # ── SIMILAR MOVIES ────────────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>⬡ You May Also Like</div>", unsafe_allow_html=True)

            recs = data['recommendations'].get('results', [])[:6]
            if recs:
                rcols = st.columns(6)
                for idx, movie in enumerate(recs):
                    with rcols[idx]:
                        if movie.get('poster_path'):
                            st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}")
                        st.caption(movie['title'])
                        if st.button("Analyse", key=f"sim_{movie['id']}"):
                            st.session_state.target_id = movie['id']
                            st.session_state.search_query = movie['title']
                            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDING NOW
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Trending Now":

    st.markdown("<p class='hero-title'>TRENDING</p>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>⬡ Live data from TMDb &nbsp;·&nbsp; Updated weekly</p>", unsafe_allow_html=True)

    try:
        trends = get_trending_movies_robust()
        if not trends:
            st.info("No trending data available at the moment.")
        else:
            for i in range(0, len(trends), 4):
                row_movies = trends[i:i+4]
                cols = st.columns(4, gap="medium")
                for idx, movie in enumerate(row_movies):
                    with cols[idx]:
                        rank = i + idx + 1
                        st.markdown(f"<div class='trend-rank'>#{rank:02d}</div>", unsafe_allow_html=True)
                        if movie.get('poster_path'):
                            st.image(f"https://image.tmdb.org/t/p/w500{movie['poster_path']}", use_container_width=True)
                        st.markdown(f"**{movie.get('title')}**")
                        st.markdown(f"<div class='trend-rating'>★ {movie.get('vote_average',0):.1f} / 10</div>", unsafe_allow_html=True)
                        if st.button("Analyse Sentiment", key=f"trend_{movie['id']}", use_container_width=True):
                            st.session_state.target_id = movie['id']
                            st.session_state.search_query = movie.get('title')
                            st.session_state.page = "Main Analytics"
                            st.rerun()
                st.markdown("<hr>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Connection failed: {e}. Please check your internet or API key.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    Anshu Raj &nbsp;·&nbsp; Semantic AI Engine &nbsp;·&nbsp; Live TMDb Integration &nbsp;
</div>
""", unsafe_allow_html=True)
