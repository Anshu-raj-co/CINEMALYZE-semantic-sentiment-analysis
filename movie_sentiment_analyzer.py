# --- Movie Review Sentiment Analysis Project ---
# 1. Data Cleaning and Pre-processing
# 2. NLP-based analysis (POS Tagging, NER)
# 3. TF-IDF Vectorization
# 4. Training and Evaluating a Machine Learning Model
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier
import joblib #saving the model


# Loading data
try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found.")
    print("Please download it from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print("And place it in the same directory as this script.")
    exit()

#lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text entry.
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    return ' '.join(processed_tokens)

print("\nStarting text pre-processing (this may take a few minutes)...")
# Applying the pre-processing function to the 'review' column
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Text pre-processing complete.")
print("\nSample of cleaned data:")
print(df[['review', 'cleaned_review', 'sentiment']].head())


#NLP-driven Insights
def get_adjectives(text):
    """Extracts adjectives from a text using POS tagging."""
    tokens = word_tokenize(text)
    
    pos_tags = pos_tag(tokens)
    adjectives = [word for word, tag in pos_tags if tag in ['JJ', 'JJR', 'JJS']]
    return adjectives

def get_named_entities(text):
    """Extracts named entities from a text using NER."""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    named_entities = []
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            entity_name = " ".join([word for word, tag in subtree.leaves()])
            named_entities.append(entity_name)
    return named_entities

# Analyze a smaller sample to get insights quickly
sample_df = df.sample(n=1000, random_state=42)

positive_reviews_sample = " ".join(sample_df[sample_df['sentiment'] == 'positive']['cleaned_review'])
negative_reviews_sample = " ".join(sample_df[sample_df['sentiment'] == 'negative']['cleaned_review'])

# Get top 10 adjectives from positive and negative reviews
top_10_positive_adj = pd.Series(get_adjectives(positive_reviews_sample)).value_counts().head(10)
top_10_negative_adj = pd.Series(get_adjectives(negative_reviews_sample)).value_counts().head(10)

print("\n--- NLP Insights ---")
print("\nTop 10 Adjectives in Positive Reviews:")
print(top_10_positive_adj)
print("\nTop 10 Adjectives in Negative Reviews:")
print(top_10_negative_adj)

# Get top 10 named entities from the entire sample
all_reviews_sample = " ".join(sample_df['review']) # Use original review for better NER
top_10_entities = pd.Series(get_named_entities(all_reviews_sample)).value_counts().head(10)

print("\nTop 10 Named Entities Mentioned:")
print(top_10_entities)


#Machine Learning Model
print("\n--- Machine Learning Model Training ---")

# Define features (X) and target (y)
X = df['cleaned_review']
y = df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Initialize TF-IDF Vectorizer
# max_features is set to 5000 to keep the feature set manageable
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit and transform the training data, transform the test data
print("\nApplying TF-IDF Vectorization...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("TF-IDF Vectorization complete.")

# Train a Logistic Regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# --- Train and Evaluate Extra Trees Classifier ---
print("\n--- Training Extra Trees Classifier ---")

# Initialize the model
# n_estimators=100 is a good starting point
# n_jobs=-1 uses all available CPU cores for training
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Training Extra Trees model (this may take a bit longer)...")
et_model.fit(X_train_tfidf, y_train)
print("Extra Trees Model training complete.")

# Make predictions on the test set
y_pred_et = et_model.predict(X_test_tfidf)

# Evaluate the Extra Trees model
accuracy_et = accuracy_score(y_test, y_pred_et)
report_et = classification_report(y_test, y_pred_et)

print("\n--- Extra Trees Model Evaluation ---")
print(f"Accuracy: {accuracy_et:.4f}")
print("\nClassification Report:")
print(report_et)

# --- Save the Extra Trees Model ---
joblib.dump(et_model, 'extra_tree_model.joblib')
print("\nExtra Trees Model has been saved as 'extra_tree_model.joblib'")

#Saving the Model and Vectorizer
# Save the trained model and the TF-IDF vectorizer for later use in a web app
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("\nModel and TF-IDF Vectorizer have been saved as 'sentiment_model.joblib' and 'tfidf_vectorizer.joblib'")


# --- How to use the model for a new review
print("\n--- Example Prediction ---")
new_review = "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
# 1. Preprocess the new review
cleaned_new_review = preprocess_text(new_review)
# 2. Transform using the fitted vectorizer
new_review_tfidf = tfidf_vectorizer.transform([cleaned_new_review])
# 3. Predict
prediction = model.predict(new_review_tfidf)
print(f"Review: '{new_review}'")
print(f"Predicted Sentiment: {prediction[0]}")

