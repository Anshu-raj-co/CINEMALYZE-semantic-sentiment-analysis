import joblib
import pandas as pd

print("⏳ Loading Vectorizer and the NEW Master Dataset...")
tfidf = joblib.load('tfidf_vectorizer.joblib')
# Load the 68k master file we just created
df = pd.read_csv('IMDB_Zenodo_Master.csv')

print(f"📊 Vectorizing {len(df)} reviews... (This may take 3-5 minutes)")
# We vectorize the 'review' column from the master file
# Use .astype(str) to prevent any last-minute type errors
review_matrix = tfidf.transform(df['review'].astype(str))

print("💾 Saving the High-Precision Semantic Index...")
joblib.dump(review_matrix, 'semantic_index.joblib')
print("✅ Done! 'semantic_index.joblib' is now aligned with your 68k master dataset.")