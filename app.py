import streamlit as st
import joblib
import pandas as pd
import requests
import re
import os
import gc
import nltk
import random
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from thefuzz import process as fuzz_process

# ── CRITICAL: set_page_config MUST be the first Streamlit command ──────────────
st.set_page_config(page_title="Cinemalyze", page_icon="🎬", layout="wide")

# ── NLTK DOWNLOADS (preserved exactly) ────────────────────────────────────────
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab',                      quiet=True)
nltk.download('stopwords',                      quiet=True)
nltk.download('punkt',                          quiet=True)
nltk.download('wordnet',                        quiet=True)
nltk.download('omw_latin1',                     quiet=True)
nltk.download('omw-1.4',                        quiet=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if 'target_id'      not in st.session_state: st.session_state.target_id      = None
if 'search_query'   not in st.session_state: st.session_state.search_query   = ""
if 'page'           not in st.session_state: st.session_state.page           = "Main Analytics"
if 'candidates'     not in st.session_state: st.session_state.candidates     = []   # Discovery grid results
if 'random_recs'    not in st.session_state: st.session_state.random_recs    = []

# ── MODEL LOADING (preserved exactly) ─────────────────────────────────────────
@st.cache_resource
def load_ai_brain():
    m_lr = joblib.load('sentiment_model.joblib')
    m_et = joblib.load('extra_tree_model.joblib')
    vec  = joblib.load('tfidf_vectorizer.joblib')
    gc.collect()
    return m_lr, m_et, vec

try:
    model_lr, model_et, tfidf_vectorizer = load_ai_brain()
except Exception as e:
    st.error(f"Error loading AI models: {e}. Ensure .joblib files are in the directory.")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "f1d69efb3938a73c4aee8a756489171d")
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"
POSTER_SM    = "https://image.tmdb.org/t/p/w200"

# ── DESIGN SYSTEM (theme-aware CSS, preserved + discovery card additions) ──────
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
    --text:    var(--text-color);
    --muted:   color-mix(in srgb, var(--text-color) 60%, transparent);
    --card:    color-mix(in srgb, var(--text-color) 5%, transparent);
    --surface: color-mix(in srgb, var(--text-color) 3%, transparent);
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main .block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }

/* ── Hero ── */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    letter-spacing: 0.06em; line-height: 1;
    background: linear-gradient(135deg, #F5C518 0%, #ffffff 50%, #87CEEB 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.hero-sub {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.25em; color: var(--muted);
    text-transform: uppercase; margin-top: 0.3rem; margin-bottom: 2rem;
}

/* ── Input ── */
[data-testid="stTextInput"] input {
    border: 1.5px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
    padding: 0.75rem 1rem !important; transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,197,24,0.12) !important;
}
[data-testid="stTextInput"] label {
    color: var(--muted) !important; font-size: 0.78rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: var(--gold) !important; color: #0a0a0f !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Bebas Neue', sans-serif !important; font-size: 1.1rem !important;
    letter-spacing: 0.1em !important; padding: 0.65rem 2.5rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(245,197,24,0.35) !important;
}
[data-testid="stButton"] > button:not([kind="primary"]) {
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.78rem !important;
    padding: 0.4rem 0.6rem !important; width: 100% !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: var(--gold) !important; background: rgba(245,197,24,0.07) !important;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--gold);
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 8px !important; border: 1px solid var(--border) !important;
    transition: transform 0.2s, border-color 0.2s !important;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.03) !important; border-color: var(--gold) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 1.25rem 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important; font-size: 0.72rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important;
    color: var(--gold) !important; letter-spacing: 0.05em !important;
}

/* ── Movie profile ── */
.movie-title-large {
    font-family: 'Bebas Neue', sans-serif; font-size: clamp(2rem, 4vw, 3.2rem);
    letter-spacing: 0.05em; color: var(--text); line-height: 1.1; margin-bottom: 0.5rem;
}
.movie-meta {
    font-family: 'DM Mono', monospace; font-size: 0.75rem;
    color: var(--muted); letter-spacing: 0.12em; margin-bottom: 1rem;
}
.genre-pill {
    display: inline-block; background: rgba(245,197,24,0.12);
    border: 1px solid rgba(245,197,24,0.3); color: var(--gold);
    border-radius: 100px; padding: 0.2rem 0.75rem; font-size: 0.72rem;
    font-family: 'DM Mono', monospace; letter-spacing: 0.08em;
    margin-right: 0.4rem; margin-bottom: 0.4rem;
}
.rating-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(245,197,24,0.1); border: 1px solid var(--gold);
    border-radius: 8px; padding: 0.3rem 0.75rem;
    font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem;
    color: var(--gold); letter-spacing: 0.05em; margin-bottom: 1rem;
}
.overview-text {
    font-size: 0.92rem; line-height: 1.7; color: var(--muted);
    border-left: 3px solid var(--gold); padding-left: 1rem; margin: 1rem 0;
}
.crew-line { font-size: 0.82rem; color: var(--muted); margin-bottom: 0.3rem; }
.crew-line span { color: var(--text); font-weight: 500; }

/* ── Discovery grid card ── */
.disc-card {
    border: 1px solid var(--border); border-radius: 12px;
    padding: 0.75rem; background: var(--card);
    transition: border-color 0.2s, transform 0.2s;
    height: 100%;
}
.disc-card:hover { border-color: var(--gold); transform: translateY(-3px); }
.disc-title {
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    font-size: 0.85rem; color: var(--text);
    margin: 0.5rem 0 0.15rem; line-height: 1.3;
}
.disc-year {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    color: var(--muted); margin-bottom: 0.5rem;
}
.disc-type {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase;
    background: rgba(245,197,24,0.1); border: 1px solid rgba(245,197,24,0.25);
    color: var(--gold); border-radius: 4px; padding: 0.1rem 0.4rem;
    margin-bottom: 0.4rem;
}

/* ── Misc ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; overflow: hidden !important;
}
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid var(--border) !important; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stCaptionContainer"] p { font-size: 0.72rem !important; color: var(--muted) !important; text-align: center; }

/* ── Trending ── */
.trend-rank {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem;
    color: rgba(245,197,24,0.15); line-height: 1; margin-bottom: -0.5rem;
}
.trend-rating { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--gold); margin-top: 0.2rem; }

/* ── Footer ── */
.footer {
    text-align: center; color: var(--muted); font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase;
    margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ── MOVIE DATABASE (curated suggestions) ──────────────────────────────────────
ALL_MOVIES_DB = [
    {"title": "The Dark Knight",       "image_url": f"{POSTER_BASE}/qJ2tW6WMUDux911r6m7haRef0WH.jpg"},
    {"title": "Inception",             "image_url": f"{POSTER_BASE}/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg"},
    {"title": "Interstellar",          "image_url": f"{POSTER_BASE}/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"},
    {"title": "Parasite",              "image_url": f"{POSTER_BASE}/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg"},
    {"title": "The Matrix",            "image_url": f"{POSTER_BASE}/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"},
    {"title": "Avatar",                "image_url": f"{POSTER_BASE}/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg"},
    {"title": "Joker",                 "image_url": f"{POSTER_BASE}/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg"},
    {"title": "Gladiator",             "image_url": f"{POSTER_BASE}/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg"},
    {"title": "Titanic",               "image_url": f"{POSTER_BASE}/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg"},
    {"title": "Avengers: Endgame",     "image_url": f"{POSTER_BASE}/or06FN3Dka5tukK1e9sl16pB3iy.jpg"},
    {"title": "Pulp Fiction",          "image_url": f"{POSTER_BASE}/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg"},
    {"title": "The Lion King",         "image_url": f"{POSTER_BASE}/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg"},
    {"title": "Spirited Away",         "image_url": f"{POSTER_BASE}/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg"},
    {"title": "Into the Spider-Verse", "image_url": f"{POSTER_BASE}/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg"},
    {"title": "Dune",                  "image_url": f"{POSTER_BASE}/d5NXSklXo0qyIYkgV94XAgMIckC.jpg"},
]

if not st.session_state.random_recs:
    st.session_state.random_recs = random.sample(ALL_MOVIES_DB, 5)

# ── NLP HELPERS ───────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def advanced_nlp_processing(text):
    clean  = re.sub(r'<.*?>', ' ', str(text))
    clean  = re.sub(r'[^a-zA-Z]', ' ', clean).lower()
    tokens = word_tokenize(clean)
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

def get_top_adjectives(texts):
    all_words = []
    for text in texts:
        tokens    = word_tokenize(str(text).lower())
        tags      = nltk.pos_tag(tokens)
        all_words.extend([w for w, t in tags
                          if t in ['JJ','JJR','JJS'] and w not in stop_words and len(w) > 2])
    return Counter(all_words)

# ── HTTP SESSION FACTORY ───────────────────────────────────────────────────────
def _session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY ENGINE  —  API-First Hybrid with Fuzzy Referee
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def _raw_multi_search(query: str) -> list:
    """
    Hit search/multi and return the raw results list.
    Falls back to a 5-char prefix search if the primary returns 0 results.
    """
    sess = _session()
    params = {"api_key": TMDB_API_KEY, "query": query, "include_adult": False}
    try:
        res     = sess.get("https://api.themoviedb.org/3/search/multi", params=params, timeout=10)
        results = res.json().get('results', [])

        # ── Fallback: typo guard — retry with first 5 chars ──────────────────
        if not results and len(query) > 5:
            params["query"] = query[:5]
            res     = sess.get("https://api.themoviedb.org/3/search/multi", params=params, timeout=10)
            results = res.json().get('results', [])

        return results
    except Exception:
        return []


@st.cache_data(ttl=3600)
def _person_known_for(person_id: int) -> list:
    """Fetch known_for movies for a person result."""
    sess = _session()
    try:
        res  = sess.get(f"https://api.themoviedb.org/3/person/{person_id}/movie_credits",
                        params={"api_key": TMDB_API_KEY}, timeout=10)
        data = res.json()
        # Return cast credits sorted by popularity
        cast = data.get('cast', [])
        cast.sort(key=lambda x: x.get('popularity', 0), reverse=True)
        return cast[:20]
    except Exception:
        return []


def discovery_search(query: str) -> list:
    """
    Full Discovery Engine pipeline.
    Returns up to 6 ranked candidate dicts ready for the selection grid.
    Each dict has: id, title, year, poster_path, media_type, overview, score
    """
    raw = _raw_multi_search(query)
    if not raw:
        return []

    movie_candidates = []
    person_hits      = []

    for r in raw:
        mtype = r.get('media_type')

        if mtype == 'person':
            # ── Person path: actor/director query (e.g. "Mom Sridevi") ───────
            person_hits.append(r)

        elif mtype in ('movie', 'tv'):
            title = r.get('title') or r.get('name') or ''
            year_raw = (r.get('release_date') or r.get('first_air_date') or '')[:4]
            movie_candidates.append({
                'id':          r.get('id'),
                'title':       title,
                'year':        year_raw,
                'poster_path': r.get('poster_path'),
                'media_type':  mtype,
                'overview':    r.get('overview', ''),
                'score':       0,
            })

    # ── Person expansion ──────────────────────────────────────────────────────
    # If any person result was found, pull their filmography and inject into candidates
    for person in person_hits[:2]:  # limit to top 2 people
        known = _person_known_for(person['id'])
        for film in known:
            title    = film.get('title', '')
            year_raw = (film.get('release_date') or '')[:4]
            movie_candidates.append({
                'id':          film.get('id'),
                'title':       title,
                'year':        year_raw,
                'poster_path': film.get('poster_path'),
                'media_type':  'movie',
                'overview':    film.get('overview', ''),
                'score':       0,
            })

    if not movie_candidates:
        return []

    # ── Fuzzy Referee: score every candidate against the user's query ─────────
    title_list = [c['title'] for c in movie_candidates]
    for cand in movie_candidates:
        _, score     = fuzz_process.extractOne(query, [cand['title']]) if cand['title'] else (None, 0)
        cand['score'] = score

    # Deduplicate by id, keeping highest score
    seen   = {}
    for c in movie_candidates:
        cid = c['id']
        if cid not in seen or c['score'] > seen[cid]['score']:
            seen[cid] = c

    ranked = sorted(seen.values(), key=lambda x: x['score'], reverse=True)
    return ranked[:6]


# ── Full details for a confirmed movie_id ─────────────────────────────────────
@st.cache_data(ttl=3600)
def get_full_movie_intelligence(movie_id: int) -> dict | None:
    sess = _session()
    try:
        res = sess.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY,
                    "append_to_response": "videos,credits,reviews,recommendations"},
            timeout=15
        )
        return res.json()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_trending_movies_robust() -> list:
    sess = _session()
    try:
        res = sess.get("https://api.themoviedb.org/3/trending/movie/week",
                       params={"api_key": TMDB_API_KEY}, timeout=10)
        return res.json().get('results', [])
    except Exception:
        return []


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem;'>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;
                    letter-spacing:0.1em;color:#F5C518;'>CINEMALYZE</div>
        <div style='font-family:"DM Mono",monospace;font-size:0.6rem;
                    letter-spacing:0.2em;color:#64748b;text-transform:uppercase;'>
            Discovery Engine · Semantic AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    page_options = ["Main Analytics", "Trending Now"]
    current_idx  = 0 if st.session_state.page == "Main Analytics" else 1
    st.session_state.page = st.sidebar.radio("Navigate to:", page_options, index=current_idx)

    st.markdown("<hr style='border-color:rgba(245,197,24,0.15);margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.65rem;
                letter-spacing:0.1em;color:#64748b;line-height:1.8;'>
        <div style='color:#F5C518;margin-bottom:0.5rem;'>MODELS ACTIVE</div>
        ▸ Logistic Regression<br>
        ▸ Extra Trees Classifier<br>
        ▸ TF-IDF Vectorizer<br><br>
        <div style='color:#F5C518;margin-bottom:0.5rem;'>SEARCH ENGINE</div>
        ▸ TMDB search/multi<br>
        ▸ Person filmography expansion<br>
        ▸ Levenshtein fuzzy ranking<br>
        ▸ 5-char typo fallback
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MAIN ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Main Analytics":

    st.markdown("<p class='hero-title'>CINEMALYZE</p>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>⬡ Discovery Engine &nbsp;·&nbsp; Semantic AI &nbsp;·&nbsp; Live Sentiment Intelligence</p>",
                unsafe_allow_html=True)

    # ── Search bar ────────────────────────────────────────────────────────────
    movie_input = st.text_input(
        "SEARCH  —  movie title, actor, director, or partial name",
        value=st.session_state.search_query,
        placeholder="search any movie, show, actor, director",
    )

    col_search, col_clear = st.columns([3, 1])
    with col_search:
        search_clicked = st.button("🔍  Discover", type="primary", use_container_width=True)
    with col_clear:
        if st.button("✕  Clear", use_container_width=True):
            st.session_state.search_query = ""
            st.session_state.candidates   = []
            st.session_state.target_id    = None
            st.rerun()

    # ── Curated suggestions ───────────────────────────────────────────────────
    if not st.session_state.candidates and not st.session_state.target_id:
        st.markdown("<div class='section-label'>⬡ All Time Classics</div>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, movie in enumerate(st.session_state.random_recs):
            with cols[i]:
                st.image(movie['image_url'], use_container_width=True)
                if st.button(movie['title'], key=f"rec_btn_{i}", use_container_width=True):
                    # Direct ID bypass — skip discovery grid for curated picks
                    st.session_state.search_query = movie['title']
                    st.session_state.candidates   = []
                    # Run quick search to get the id
                    hits = discovery_search(movie['title'])
                    if hits:
                        st.session_state.target_id = hits[0]['id']
                    st.rerun()

    # ── Run discovery when search clicked ────────────────────────────────────
    if search_clicked and movie_input.strip():
        st.session_state.search_query = movie_input.strip()
        st.session_state.target_id    = None   # reset any previously locked film
        with st.spinner("Running Discovery Engine…"):
            st.session_state.candidates = discovery_search(movie_input.strip())
        if not st.session_state.candidates:
            st.error(f"No results found for **'{movie_input}'**. Try a different spelling or add more context.")

    # ══════════════════════════════════════════════════════════════════════════
    # SELECTION GRID  (Netflix-style 3-col, top 6 results)
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.candidates and not st.session_state.target_id:
        st.markdown("<div class='section-label'>⬡ Discovery Results — Select a Title to Analyse</div>",
                    unsafe_allow_html=True)

        candidates = st.session_state.candidates
        COLS       = 3
        PLACEHOLDER = "https://via.placeholder.com/300x450/1a1a2e/F5C518?text=No+Poster"

        for row_start in range(0, len(candidates), COLS):
            row = candidates[row_start : row_start + COLS]
            cols = st.columns(COLS, gap="medium")

            for col_idx, cand in enumerate(row):
                with cols[col_idx]:
                    # Poster
                    poster_url = (f"{POSTER_BASE}{cand['poster_path']}"
                                  if cand.get('poster_path') else PLACEHOLDER)
                    st.image(poster_url, use_container_width=True)

                    # Metadata
                    mtype_label = "TV Show" if cand['media_type'] == 'tv' else "Movie"
                    year_label  = cand['year'] if cand['year'] else "—"
                    match_pct   = cand['score']

                    st.markdown(f"""
                    <div class='disc-type'>{mtype_label}</div>
                    <div class='disc-title'>{cand['title']}</div>
                    <div class='disc-year'>{year_label} &nbsp;·&nbsp; Match: {match_pct}%</div>
                    """, unsafe_allow_html=True)

                    if st.button("▶ Analyse", key=f"disc_{cand['id']}_{row_start}_{col_idx}",
                                 use_container_width=True):
                        st.session_state.target_id  = cand['id']
                        st.session_state.candidates = []   # collapse grid
                        st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FULL ANALYSIS  (fires once target_id is set)
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.target_id:

        with st.spinner("Fetching full intelligence…"):
            data = get_full_movie_intelligence(st.session_state.target_id)

        if not data:
            st.error("📡 Connection error. Please check your network or API key settings.")
        else:
            # ── Film Profile ──────────────────────────────────────────────────
            st.markdown("<div class='section-label'>⬡ Film Profile</div>", unsafe_allow_html=True)

            director   = next((p['name'] for p in data.get('credits', {}).get('crew', [])
                               if p['job'] == 'Director'), "N/A")
            cast_names = ', '.join([p['name'] for p in data.get('credits', {}).get('cast', [])[:5]])
            genres_html= ''.join([f"<span class='genre-pill'>{g['name']}</span>"
                                  for g in data.get('genres', [])])
            trailers   = [v for v in data.get('videos', {}).get('results', [])
                          if v['type'] == 'Trailer' and v['site'] == 'YouTube']
            release    = (data.get('release_date') or '')[:4]

            c1, c2 = st.columns([1, 2.8], gap="large")
            with c1:
                if data.get('poster_path'):
                    st.image(f"{POSTER_BASE}{data['poster_path']}", use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class='movie-title-large'>{data.get('title','—')}</div>
                <div class='movie-meta'>
                    {release} &nbsp;·&nbsp; {data.get('runtime','—')} min
                    &nbsp;·&nbsp; {data.get('original_language','').upper()}
                </div>
                <div style='margin-bottom:1rem;'>{genres_html}</div>
                <div class='rating-badge'>★ &nbsp;{data.get('vote_average', 0):.1f}
                    <span style='font-family:"DM Sans",sans-serif;font-size:0.75rem;
                                 color:var(--muted);margin-left:0.4rem;'>
                        /10 &nbsp;({data.get('vote_count','—')} votes)
                    </span>
                </div>
                <div class='overview-text'>{data.get('overview','')}</div>
                <div class='crew-line'>🎬 Director &nbsp;<span>{director}</span></div>
                <div class='crew-line'>🎭 Cast &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>{cast_names}</span></div>
                """, unsafe_allow_html=True)

                if trailers:
                    with st.expander("▶  Watch Trailer"):
                        st.video(f"https://www.youtube.com/watch?v={trailers[0]['key']}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Sentiment Engine ──────────────────────────────────────────────
            reviews = data.get('reviews', {}).get('results', [])

            if not reviews:
                st.warning("No user reviews found on TMDb for this title.")
            else:
                st.markdown("<div class='section-label'>⬡ AI Sentiment Analysis</div>",
                            unsafe_allow_html=True)

                table_data = []
                pos_lr, pos_et = 0, 0
                p_content, n_content = [], []

                # ── BALANCED: analyse up to 50 reviews (bias reduction) ───────
                for r in reviews[:50]:
                    clean  = advanced_nlp_processing(r['content'])
                    vec    = tfidf_vectorizer.transform([clean])
                    p_lr   = model_lr.predict(vec)[0]
                    p_et   = model_et.predict(vec)[0]
                    conf   = f"{max(model_lr.predict_proba(vec)[0])*100:.1f}%"

                    if p_lr == 'positive':
                        pos_lr += 1
                        p_content.append(r['content'])
                    else:
                        n_content.append(r['content'])
                    if p_et == 'positive':
                        pos_et += 1

                    table_data.append({
                        "Author":          r['author'],
                        "Log. Regression": p_lr.upper(),
                        "Extra Trees":     p_et.upper(),
                        "Confidence":      conf,
                        "Full Review":     r['content'],
                    })

                total = len(table_data)

                # ── Metrics row ───────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Reviews Analysed",  total)
                m2.metric("LR Positive",       f"{(pos_lr/total)*100:.0f}%")
                m3.metric("ET Positive",        f"{(pos_et/total)*100:.0f}%")
                m4.metric("Model Agreement",   "High ✓" if abs(pos_lr - pos_et) <= 1 else "Varying")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── PUBLIC vs AI COMPARISON CHART (new) ──────────────────────
                st.markdown("<div class='section-label'>⬡ Public Rating vs AI Sentiment</div>",
                            unsafe_allow_html=True)

                public_score = round((data.get('vote_average', 0) / 10) * 100, 1)
                lr_score     = round((pos_lr / total) * 100, 1)
                et_score     = round((pos_et / total) * 100, 1)

                bar_fig = go.Figure()
                bar_fig.add_trace(go.Bar(
                    name="Public Rating (TMDb)",
                    x=["Public Rating (TMDb)", "AI — Logistic Regression", "AI — Extra Trees"],
                    y=[public_score, lr_score, et_score],
                    marker_color=["#F5C518", "#87CEEB", "#4FA3C7"],
                    text=[f"{v}%" for v in [public_score, lr_score, et_score]],
                    textposition='outside',
                    textfont=dict(family="DM Mono", size=13, color="#F5C518"),
                    hovertemplate="%{x}<br>Score: %{y}%<extra></extra>",
                    showlegend=False,
                    width=0.45,
                ))
                bar_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="DM Sans", color="#94a3b8"),
                    margin=dict(t=30, b=10, l=10, r=10),
                    yaxis=dict(
                        range=[0, 115],
                        ticksuffix="%",
                        gridcolor="rgba(245,197,24,0.08)",
                        zeroline=False,
                    ),
                    xaxis=dict(tickfont=dict(size=12)),
                    bargap=0.35,
                )
                st.plotly_chart(bar_fig, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Donut + Word Traits ───────────────────────────────────────
                v1, v2 = st.columns([1.2, 1], gap="large")
                with v1:
                    donut = go.Figure(go.Pie(
                        labels=["Positive", "Negative"],
                        values=[pos_lr, total - pos_lr],
                        hole=0.55,
                        marker_colors=["#87CEEB", "#E50914"],
                        textinfo='label+percent',
                        textfont=dict(family="DM Mono", size=12, color="white"),
                        hovertemplate="%{label}: %{value} reviews<extra></extra>",
                    ))
                    donut.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                        annotations=[dict(
                            text=f"<b>{(pos_lr/total)*100:.0f}%</b><br>positive",
                            x=0.5, y=0.5, font_size=18,
                            font_color="#F5C518", font_family="Bebas Neue", showarrow=False,
                        )],
                    )
                    st.plotly_chart(donut, use_container_width=True)

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

                # ── Review table ──────────────────────────────────────────────
                st.markdown("<div class='section-label' style='margin-top:1.5rem;'>⬡ Review Breakdown</div>",
                            unsafe_allow_html=True)
                df = pd.DataFrame(table_data)

                def style_sent(val):
                    if val == 'POSITIVE': return 'color: #87CEEB; font-weight: 600;'
                    if val == 'NEGATIVE': return 'color: #E50914; font-weight: 600;'
                    return ''

                st.dataframe(
                    df.style.map(style_sent, subset=['Log. Regression', 'Extra Trees']),
                    use_container_width=True, hide_index=True,
                )

            # ── Similar Movies ─────────────────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>⬡ You May Also Like</div>", unsafe_allow_html=True)

            recs = data.get('recommendations', {}).get('results', [])[:6]
            if recs:
                rcols = st.columns(6)
                for idx, movie in enumerate(recs):
                    with rcols[idx]:
                        if movie.get('poster_path'):
                            st.image(f"{POSTER_SM}{movie['poster_path']}")
                        st.caption(movie.get('title',''))
                        if st.button("Analyse", key=f"sim_{movie['id']}"):
                            st.session_state.target_id  = movie['id']
                            st.session_state.candidates = []
                            st.session_state.search_query = movie.get('title','')
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDING NOW
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Trending Now":

    st.markdown("<p class='hero-title'>TRENDING</p>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>⬡ Live data from TMDb &nbsp;·&nbsp; Updated weekly</p>",
                unsafe_allow_html=True)

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
                            st.image(f"{POSTER_BASE}{movie['poster_path']}", use_container_width=True)
                        st.markdown(f"**{movie.get('title', movie.get('name',''))}**")
                        st.markdown(f"<div class='trend-rating'>★ {movie.get('vote_average',0):.1f} / 10</div>",
                                    unsafe_allow_html=True)
                        if st.button("Analyse Sentiment", key=f"trend_{movie['id']}", use_container_width=True):
                            st.session_state.target_id    = movie['id']
                            st.session_state.search_query = movie.get('title', movie.get('name',''))
                            st.session_state.candidates   = []
                            st.session_state.page         = "Main Analytics"
                            st.rerun()
                st.markdown("<hr>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Connection failed: {e}. Please check your network or API key settings.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    Anshu Raj &nbsp;·&nbsp; Discovery Engine &nbsp;·&nbsp; Semantic AI &nbsp;·&nbsp; Live TMDb Integration
</div>
""", unsafe_allow_html=True)
