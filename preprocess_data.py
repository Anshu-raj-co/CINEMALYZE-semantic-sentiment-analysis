import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 1. Download requirements
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 2. Load the original 50k dataset
df = pd.read_csv('IMDB Dataset.csv')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_nlp_processing(text):
    text_clean = re.sub(r'<.*?>', ' ', str(text))
    text_clean = re.sub(r'[^a-zA-Z]', ' ', text_clean).lower()
    tokens = word_tokenize(text_clean)
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

print("⏳ Cleaning 50,000 reviews... this will take 2-3 minutes.")
df['cleaned_review'] = df['review'].apply(advanced_nlp_processing)

# 3. Save as a NEW file
df.to_csv('IMDB_Preprocessed.csv', index=False)
print("✅ Done! 'IMDB_Preprocessed.csv' has been created.")