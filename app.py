import streamlit as st
import joblib
import re
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from urllib.parse import quote
import random
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from collections import Counter

# --- 0. NLTK Resource Initialization ---
@st.cache_resource
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger']
    for res in resources:
        nltk.download(res, quiet=True)

download_nltk_resources()

# --- 1. CONFIGURATION & UI STYLING ---
OMDB_API_KEY = "d21f0838"
st.set_page_config(page_title="Cinemalyze", page_icon="🎬", layout="wide")

# Custom CSS Preserved
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&family=Inter:wght@400&display=swap');
    :root { --color-primary: #14f195; --color-bg: #1e293b; --color-text: #cbd5e1; }
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; color: var(--color-text); }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; color: white; }
    .stMetric { background-color: #334155; border-radius: 12px; padding: 15px; border: 1px solid #475569; }
    .footer { text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 50px; }
    .theme-tag { background-color: #14f195; color: #1e293b; padding: 4px 12px; border-radius: 15px; margin-right: 8px; display: inline-block; font-weight: bold; font-size: 0.8rem; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# Trending Suggestions Preserved
ALL_MOVIES_DB = [
    {"title": "The Dark Knight", "image_url": "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg"},
    {"title": "Inception", "image_url": "https://upload.wikimedia.org/wikipedia/en/2/2e/Inception_%282010%29_theatrical_poster.jpg"},
    {"title": "Interstellar", "image_url": "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg"},
    {"title": "Parasite", "image_url": "https://upload.wikimedia.org/wikipedia/en/5/53/Parasite_%282019_film%29.png"},
    {"title": "Sultan", "image_url": "https://upload.wikimedia.org/wikipedia/en/1/1f/Sultan_film_poster.jpg"},
    {"title": "The Matrix", "image_url": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"},
    {"title": "Avatar", "image_url": "https://upload.wikimedia.org/wikipedia/en/d/d6/Avatar_%282009_film%29_poster.jpg"},
    {"title": "Joker", "image_url": "https://upload.wikimedia.org/wikipedia/en/e/e1/Joker_%282019_film%29_poster.jpg"},
    {"title": "Gladiator", "image_url": "https://upload.wikimedia.org/wikipedia/en/f/fb/Gladiator_%282000_film_poster%29.png"},
    {"title": "Titanic", "image_url": "https://upload.wikimedia.org/wikipedia/en/1/18/Titanic_%281997_film%29_poster.png"},
    {"title": "Avengers: Endgame", "image_url": "https://upload.wikimedia.org/wikipedia/en/0/0d/Avengers_Endgame_poster.jpg"},
    {"title": "The Godfather", "image_url": "https://upload.wikimedia.org/wikipedia/en/1/1c/Godfather_ver1.jpg"}
]

if 'random_recs' not in st.session_state:
    st.session_state.random_recs = random.sample(ALL_MOVIES_DB, 5)

# --- 2. NLP Pipeline ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_nlp_processing(text):
    text_clean = re.sub(r'<.*?>', ' ', str(text))
    text_clean = re.sub(r'[^a-zA-Z]', ' ', text_clean).lower()
    tokens = word_tokenize(text_clean)
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

def get_top_adjectives(texts, n=5):
    all_words = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        tags = nltk.pos_tag(tokens)
        adjectives = [w for w, t in tags if t in ['JJ', 'JJR', 'JJS'] and w not in stop_words and len(w) > 2]
        all_words.extend(adjectives)
    return [item[0] for item in Counter(all_words).most_common(n)]

# --- 3. Asset Loading (INSTANT LOAD VERSION) ---
@st.cache_resource
def load_all_assets():
    try:
        model_lr = joblib.load('sentiment_model.joblib')
        model_et = joblib.load('extra_tree_model.joblib')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
        df = pd.read_csv('IMDB_Preprocessed.csv')
        
        # Load the pre-calculated matrix (The Instant Fix)
        review_matrix = joblib.load('semantic_index.joblib')
        
        with st.spinner("Quick-Syncing Performance Metrics..."):
            sample_df = df.sample(min(400, len(df))) 
            sample_vectors = tfidf.transform(sample_df['cleaned_review'].fillna(''))
            acc_lr = f"{accuracy_score(sample_df['sentiment'].str.lower(), model_lr.predict(sample_vectors))*100:.1f}%"
            acc_et = f"{accuracy_score(sample_df['sentiment'].str.lower(), model_et.predict(sample_vectors))*100:.1f}%"

        return model_lr, model_et, tfidf, df, review_matrix, acc_lr, acc_et
    except Exception as e:
        st.error(f"Launch Error: Ensure 'semantic_index.joblib' exists. Error: {e}")
        st.stop()

model_lr, model_et, tfidf_vectorizer, local_df, global_review_matrix, acc_lr, acc_et = load_all_assets()

# --- 4. Analytics Engine ---
@st.cache_data(ttl=3600)
def get_movie_analysis(movie_name):
    api_url = f"http://www.omdbapi.com/?t={quote(movie_name)}&apikey={OMDB_API_KEY}&plot=full"
    try:
        res = requests.get(api_url, timeout=10).json()
        if res.get('Response') == 'False':
            return None, None, None, None, res.get('Error', 'Movie not found.')

        query_text = advanced_nlp_processing(res.get('Plot', '') + " " + res.get('Genre', ''))
        query_vec = tfidf_vectorizer.transform([query_text])
        
        # Keyword Extraction via TF-IDF Importance
        feature_names = tfidf_vectorizer.get_feature_names_out()
        query_scores = query_vec.toarray()[0]
        top_keyword_indices = query_scores.argsort()[-6:][::-1]
        keywords = [feature_names[i] for i in top_keyword_indices if query_scores[i] > 0]
        
        similarities = cosine_similarity(query_vec, global_review_matrix).flatten()
        
        # Normalize and filter (0.30 threshold preserved)
        max_s = max(similarities) if max(similarities) > 0 else 1
        normalized_similarities = similarities / max_s
        threshold = 0.30
        sorted_indices = normalized_similarities.argsort()[::-1]
        
        final_indices = []
        for i in sorted_indices:
            if normalized_similarities[i] > threshold:
                final_indices.append(i)
            if len(final_indices) == 10: break
        
        reviews = local_df.iloc[final_indices]['review'].tolist()
        scores = normalized_similarities[final_indices].tolist()

        return res, reviews, scores, keywords, None
    except Exception as e:
        return None, None, None, None, str(e)

# --- 5. UI Layout (Exactly the same as before) ---
st.title("🎬 Cinemalyze")
st.markdown("### Semantic AI & Model Comparison Dashboard")

tab1, tab2, tab3 = st.tabs(["🚀 Main Analytics", "📂 Full Database Search", "✍️ Manual Test"])

with tab1:
    movie_q = st.text_input("Enter Movie Title", placeholder="e.g. Sultan, Interstellar")
    
    st.markdown("<h4 style='color: var(--color-text-muted);'>Try a suggested film:</h4>", unsafe_allow_html=True)
    cols = st.columns(5)
    selected_rec = None
    for i, movie in enumerate(st.session_state.random_recs):
        with cols[i]:
            st.image(movie['image_url'], use_container_width=True)
            if st.button(movie['title'], key=f"rec_btn_{i}", use_container_width=True):
                selected_rec = movie['title']

    target_movie = selected_rec if selected_rec else movie_q
    if (st.button("Run Engine", type="primary") or selected_rec) and target_movie:
        with st.spinner(f"Computing semantic vectors..."):
            movie_data, reviews, scores, keywords, error = get_movie_analysis(target_movie)
            
            if movie_data:
                # --- MOVIE HEADER SECTION ---
                c1, c2 = st.columns([1, 2])
                with c1:
                    if movie_data.get('Poster') != "N/A": st.image(movie_data['Poster'])
                with c2:
                    st.header(f"{movie_data['Title']} ({movie_data['Year']})")
                    # Detected Themes preserved
                    st.markdown("### 🔍 Detected Themes")
                    theme_html = "".join([f'<span class="theme-tag">{k.upper()}</span>' for k in keywords])
                    st.markdown(theme_html, unsafe_allow_html=True)
                    st.write(f"**Plot Summary:** {movie_data['Plot']}")
                
                st.divider()

                if not reviews:
                    st.warning("Low thematic similarity detected (Below 0.30 threshold).")
                else:
                    # --- SENTIMENT CALCULATION LOOP ---
                    pos_list, neg_list = [], []
                    results_table = []
                    
                    for r, score in zip(reviews, scores):
                        clean_r = advanced_nlp_processing(r)
                        vec_r = tfidf_vectorizer.transform([clean_r])
                        pred = model_lr.predict(vec_r)[0]
                        prob = model_lr.predict_proba(vec_r)[0]
                        
                        if pred == 'positive': pos_list.append(r)
                        else: neg_list.append(r)

                        results_table.append({
                            "Relevant Review": r.replace('<br /><br />', ' '), 
                            "Sentiment": pred.upper(),
                            "Rel. Similarity": f"{score:.2f}",
                            "Confidence": f"{max(prob)*100:.1f}%" 
                        })

                    # --- HERO METRICS (Requirement Fix: Model Comparison vs IMDb) ---
                    st.markdown("### 🤖 High-Level Intelligence & Model Comparison")
                    h1, h2, h3, h4 = st.columns(4)

                    # 1. IMDb Rating (Direct from OMDb API)
                    imdb_val = movie_data.get('imdbRating', 'N/A')
                    h1.metric("IMDb Rating", f"⭐ {imdb_val}")

                    # 2. Logistic Regression Performance
                    h2.metric("Log. Regression Accuracy", acc_lr)

                    # 3. Extra Trees Performance
                    h3.metric("Extra Trees Accuracy", acc_et)

                    # 4. Data Relevance
                    h4.metric("Thematic Matches", f"{len(reviews)} Reviews")

                    st.divider()

                    # --- DETAILED INSIGHTS & CHARTS ---
                    st.markdown("### 🧠 Semantic NLP Insights")
                    i1, i2 = st.columns(2)
                    with i1:
                        st.write("🟢 **Positive Sentiment Indicators:**")
                        st.write(", ".join(get_top_adjectives(pos_list)) if pos_list else "N/A")
                    with i2:
                        st.write("🔴 **Negative Sentiment Indicators:**")
                        st.write(", ".join(get_top_adjectives(neg_list)) if neg_list else "N/A")

                    st.divider()

                    # Chart and Table preserved
                    met_col, chart_col = st.columns([1, 2])
                    with chart_col:
                        fig = px.pie(names=["Positive", "Negative"], values=[len(pos_list), len(neg_list)], hole=0.5,
                                     color=["Positive", "Negative"], color_discrete_map={"Positive": "#14f195", "Negative": "#ff4b4b"})
                        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

            else: st.error(f"Error: {error}")
# Tab 2 & 3 preserved
with tab2:
    st.header("Global Search")
    sq = st.text_input("Filter database by keyword:")
    if sq:
        st.dataframe(local_df[local_df['review'].str.contains(sq, case=False)].head(15))

with tab3:
    st.header("Manual Sentiment Test")
    user_txt = st.text_area("Analyze custom movie feedback:", height=150)
    if st.button("Predict"):
        v = tfidf_vectorizer.transform([advanced_nlp_processing(user_txt)])
        st.write(f"Prediction: **{model_lr.predict(v)[0].upper()}**")

st.markdown('<div class="footer">Built by Anshu Raj • Semantic AI • ML Pipeline project</div>', unsafe_allow_html=True)