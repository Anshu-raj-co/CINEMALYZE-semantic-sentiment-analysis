import joblib
import pandas as pd

print("⏳ Loading Vectorizer and Preprocessed Data...")
tfidf = joblib.load('tfidf_vectorizer.joblib')
df = pd.read_csv('IMDB_Preprocessed.csv')

print("📊 Vectorizing 50,000 reviews (The 'Heavy' Part)...")
# We do the 4-minute wait ONE LAST TIME here.
review_matrix = tfidf.transform(df['cleaned_review'].fillna(''))

print("💾 Saving Semantic Index...")
joblib.dump(review_matrix, 'semantic_index.joblib')
print("✅ Done! You now have 'semantic_index.joblib'. Your app will now load instantly.")