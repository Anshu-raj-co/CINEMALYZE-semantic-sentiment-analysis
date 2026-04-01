import streamlit as st
import joblib
import re
import os
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

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "d21f0838")
st.set_page_config(page_title="Cinemalyze", page_icon="🎬", layout="wide")

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

# Movie DB preserved
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

def get_top_adjectives(texts, n=15):
    all_words = []
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        tags = nltk.pos_tag(tokens)
        adjectives = [w for w, t in tags if t in ['JJ', 'JJR', 'JJS'] and w not in stop_words and len(w) > 2]
        all_words.extend(adjectives)
    return Counter(all_words)

# SSR Filter Fix (Great is no longer negative)
def get_exclusive_adjectives(pos_reviews, neg_reviews, n=5):
    pos_counts = get_top_adjectives(pos_reviews)
    neg_counts = get_top_adjectives(neg_reviews)
    exclusive_pos, exclusive_neg = [], []
    for word, count in pos_counts.most_common(50):
        if count > neg_counts.get(word, 0) * 1.5: exclusive_pos.append(word)
    for word, count in neg_counts.most_common(50):
        if count > pos_counts.get(word, 0) * 1.5: exclusive_neg.append(word)
    return exclusive_pos[:n], exclusive_neg[:n]

# --- 3. Asset Loading (Type-Safe Fix) ---
@st.cache_resource
def load_all_assets():
    try:
        model_lr = joblib.load('sentiment_model.joblib')
        model_et = joblib.load('extra_tree_model.joblib')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
        
        # Defensive Loading
        df = pd.read_csv('IMDB_Zenodo_Master.csv', low_memory=False)
        for col in ['sentiment', 'review', 'movie_title', 'source']:
            df[col] = df[col].fillna('unknown').astype(str)
            
        review_matrix = joblib.load('semantic_index.joblib')
        
        with st.spinner("Syncing Metrics..."):
            valid_rows = df[df['sentiment'].str.lower().isin(['positive', 'negative'])]
            sample_df = valid_rows.sample(min(400, len(valid_rows))) 
            sample_vectors = tfidf.transform(sample_df['review'])
            
            y_true = sample_df['sentiment'].str.lower().tolist()
            acc_lr = f"{accuracy_score(y_true, model_lr.predict(sample_vectors))*100:.1f}%"
            acc_et = f"{accuracy_score(y_true, model_et.predict(sample_vectors))*100:.1f}%"

        return model_lr, model_et, tfidf, df, review_matrix, acc_lr, acc_et
    except Exception as e:
        st.error(f"Launch Error: {e}")
        st.stop()

model_lr, model_et, tfidf_vectorizer, local_df, global_review_matrix, acc_lr, acc_et = load_all_assets()

# --- 4. Analytics Engine (Entity-Boosted) ---
@st.cache_data(ttl=3600)
def get_movie_analysis(movie_name):
    api_url = f"http://www.omdbapi.com/?t={quote(movie_name)}&apikey={OMDB_API_KEY}&plot=full"
    try:
        res = requests.get(api_url, timeout=10).json()
        if res.get('Response') == 'False': return None, None, None, None, "Movie not found."

        target_title = res.get('Title')
        director = res.get('Director', '').split(',')[0] 
        actors = res.get('Actors', '').split(', ')[:2]   

        # 1. VERIFIED SEARCH
        direct_matches = local_df[(local_df['source'] == 'Verified') & 
                                  (local_df['movie_title'].str.lower() == target_title.lower())]
        
        if len(direct_matches) >= 3:
            reviews = direct_matches['review'].head(10).tolist()
            scores = [1.0] * len(reviews)
            match_type = "Verified Title Match"
        else:
            # 2. ENTITY-BOOSTED FALLBACK
            boosted_query = f"{director} " + " ".join(actors) + " " + res.get('Plot', '')
            query_text = advanced_nlp_processing(boosted_query)
            query_vec = tfidf_vectorizer.transform([query_text])
            
            similarities = cosine_similarity(query_vec, global_review_matrix).flatten()
            max_s = max(similarities) if max(similarities) > 0 else 1
            norm_sims = similarities / max_s
            
            sorted_indices = norm_sims.argsort()[::-1]
            final_indices = []
            for i in sorted_indices:
                if local_df.iloc[i]['source'] == 'Thematic' and norm_sims[i] > 0.25:
                    final_indices.append(i)
                if len(final_indices) == 10: break
            
            reviews = local_df.iloc[final_indices]['review'].tolist()
            scores = norm_sims[final_indices].tolist()
            match_type = "Entity-Boosted Semantic Match"

        # Theme Extraction
        feature_names = tfidf_vectorizer.get_feature_names_out()
        query_scores = tfidf_vectorizer.transform([advanced_nlp_processing(res.get('Plot'))]).toarray()[0]
        keywords = [feature_names[i] for i in query_scores.argsort()[-6:][::-1] if query_scores[i] > 0]

        return res, reviews, scores, keywords, match_type
    except Exception as e: return None, None, None, None, str(e)

# --- 5. UI Layout ---
st.title("🎬 Cinemalyze")
st.markdown("### Semantic AI & Model Comparison Dashboard")

tab1, tab2, tab3 = st.tabs(["🚀 Main Analytics", "📂 Full Database Search", "✍️ Manual Test"])

with tab1:
    movie_q = st.text_input("Enter Movie Title", placeholder="e.g. Iron Man, Interstellar")
    
    st.markdown("<h4 style='color: var(--color-text-muted);'>Try a suggested film:</h4>", unsafe_allow_html=True)
    cols = st.columns(5)
    selected_rec = None
    for i, movie in enumerate(st.session_state.random_recs):
        with cols[i]:
            st.image(movie['image_url'], use_container_width=True)
            if st.button(movie['title'], key=f"rec_btn_{i}", use_container_width=True): selected_rec = movie['title']

    target_movie = selected_rec if selected_rec else movie_q
    if (st.button("Run Engine", type="primary") or selected_rec) and target_movie:
        with st.spinner(f"Processing..."):
            movie_data, reviews, scores, keywords, match_type = get_movie_analysis(target_movie)
            
            if movie_data:
                st.markdown(f"### 🤖 High-Level Intelligence | {match_type}")
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("IMDb Rating", f"⭐ {movie_data.get('imdbRating', 'N/A')}")
                h2.metric("Log. Regression Accuracy", acc_lr)
                h3.metric("Extra Trees Accuracy", acc_et)
                h4.metric("Reliability", "High" if match_type == "Verified Title Match" else "Medium")

                c1, c2 = st.columns([1, 2])
                with c1:
                    if movie_data.get('Poster') != "N/A": st.image(movie_data['Poster'])
                with c2:
                    st.header(f"{movie_data['Title']} ({movie_data['Year']})")
                    theme_html = "".join([f'<span class="theme-tag">{k.upper()}</span>' for k in keywords])
                    st.markdown(theme_html, unsafe_allow_html=True)
                    st.write(f"**Plot Summary:** {movie_data['Plot']}")
                
                st.divider()

                if not reviews:
                    st.warning("Low thematic similarity — No reviews met the quality threshold.")
                else:
                    pos_list, neg_list, results_table = [], [], []
                    for r, score in zip(reviews, scores):
                        clean_r = advanced_nlp_processing(r)
                        vec_r = tfidf_vectorizer.transform([clean_r])
                        pred = model_lr.predict(vec_r)[0]
                        prob = model_lr.predict_proba(vec_r)[0]
                        if pred == 'positive': pos_list.append(r)
                        else: neg_list.append(r)

                        results_table.append({
                            "Source": "✅ Verified" if score == 1.0 else "🔍 Thematic",
                            "Relevant Review": r.replace('<br /><br />', ' '), 
                            "Sentiment": pred.upper(),
                            "Confidence": f"{max(prob)*100:.1f}%" 
                        })

                    st.markdown("### 🧠 Semantic NLP Insights")
                    i1, i2 = st.columns(2)
                    p_adj, n_adj = get_exclusive_adjectives(pos_list, neg_list)
                    with i1:
                        st.write("🟢 **Positive Sentiment Indicators:**")
                        st.write(", ".join(p_adj) if p_adj else "N/A")
                    with i2:
                        st.write("🔴 **Negative Sentiment Indicators:**")
                        st.write(", ".join(n_adj) if n_adj else "N/A")

                    st.divider()
                    met_col, chart_col = st.columns([1, 2])
                    pos_count = len(pos_list)
                    with met_col:
                        st.metric("Final Sentiment Score", f"{(pos_count/len(reviews))*100:.1f}% Positive")
                    with chart_col:
                        fig = px.pie(names=["Positive", "Negative"], values=[pos_count, len(reviews)-pos_count], hole=0.5,
                                     color=["Positive", "Negative"], color_discrete_map={"Positive": "#14f195", "Negative": "#ff4b4b"})
                        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    df_res = pd.DataFrame(results_table)
                    def color_sent(val):
                        if val == 'POSITIVE': return 'color: #14f195;'
                        elif val == 'NEGATIVE': return 'color: #ff4b4b;'
                        return ''
                    st.dataframe(df_res.style.applymap(color_sent, subset=['Sentiment']), use_container_width=True, hide_index=True)

with tab2:
    st.header("Global Search")
    sq = st.text_input("Filter database by keyword:")
    if sq: st.dataframe(local_df[local_df['review'].str.contains(sq, case=False)].head(15))

with tab3:
    st.header("Manual Sentiment Test")
    user_txt = st.text_area("Analyze custom movie feedback:", height=150)
    if st.button("Predict"):
        v = tfidf_vectorizer.transform([advanced_nlp_processing(user_txt)])
        st.write(f"Prediction: **{model_lr.predict(v)[0].upper()}**")

st.markdown('<div class="footer">Anshu Raj • Semantic AI v4.0 • Aligned & Entity-Boosted</div>', unsafe_allow_html=True)