import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv(r"C:\Cinemalyze\PROJECT\IMDB Dataset.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Load vectorizer
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Load models
lr_model = joblib.load("sentiment_model.joblib")
et_model = joblib.load("extra_tree_model.joblib")

# Predictions
lr_pred = lr_model.predict(X_test_vec)
et_pred = et_model.predict(X_test_vec)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Extra Trees Accuracy:", accuracy_score(y_test, et_pred))

# Detailed report
print("\nLogistic Regression Report:\n", classification_report(y_test, lr_pred))
print("\nExtra Trees Report:\n", classification_report(y_test, et_pred))