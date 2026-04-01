import pandas as pd

# 1. Load your datasets (Use the files you uploaded)
# If you have the preprocessed IMDB file, use that. Otherwise, use the raw one.
imdb = pd.read_csv('IMDB Dataset.csv') 
zenodo = pd.read_csv('movie_reviews.csv') #

# 2. Clean Zenodo (The "Verified" Source)
# FIX: Drop duplicates and shuffle to ensure variety in reviews
zenodo = zenodo.drop_duplicates(subset=['review_text']).sample(frac=1).reset_index(drop=True) #
zenodo_cleaned = zenodo[['movie_name', 'review_text', 'directors', 'actors']].copy()
zenodo_cleaned.columns = ['movie_title', 'review', 'director', 'actors']

# FORCE STRING TYPE to prevent the float vs str error
for col in zenodo_cleaned.columns:
    zenodo_cleaned[col] = zenodo_cleaned[col].fillna('Unknown').astype(str)

zenodo_cleaned['source'] = 'Verified'
zenodo_cleaned['sentiment'] = 'unknown'

# 3. Prepare IMDB (The "Thematic" Source)
imdb['review'] = imdb['review'].fillna('Unknown').astype(str)
imdb['sentiment'] = imdb['sentiment'].fillna('unknown').astype(str)
imdb['movie_title'] = 'Unknown'
imdb['source'] = 'Thematic'
imdb['director'] = 'Unknown'
imdb['actors'] = 'Unknown'

# 4. Combine and Save
master_df = pd.concat([zenodo_cleaned, imdb], ignore_index=True)
master_df.to_csv('IMDB_Zenodo_Master.csv', index=False)
print("✅ Master Dataset FIXED: Deduplicated, Shuffled, and Type-Safe!")