---
title: Cinemalyze
emoji: 🎬
colorFrom: yellow
colorTo: blue
sdk: streamlit
sdk_version: 1.43.0
app_file: app.py
pinned: false
---
# 🎬 Cinemalyze — Real-Time Movie Sentiment Intelligence Engine

Cinemalyze is a real-time NLP-powered movie intelligence platform that analyzes audience sentiment using live data from TMDb. It combines dual-model machine learning (Logistic Regression + Extra Trees) with advanced text processing to deliver interactive, explainable insights into movie reviews.

🔗 **Live Demo:** https://anshuraj1212-cinemalyze.hf.space/

---

##  Key Features

*  **Real-Time Sentiment Analysis** — Fetches live user reviews using TMDb API and performs instant sentiment prediction
*  **Dual-Model Inference** — Combines Logistic Regression (efficiency) and Extra Trees (robustness) for reliable predictions
*  **Advanced NLP Pipeline** — Tokenization, stopword removal, lemmatization, and TF-IDF vectorization
*  **Interactive Dashboard** — Visual insights using Plotly (sentiment distribution, metrics, trends)
*  **Linguistic Insights** — Extracts key adjectives from reviews using POS tagging
*  **Model Agreement Score** — Compares predictions across models for consistency
*  **Movie Intelligence Layer** — Displays movie metadata, cast, trailer, and recommendations

---

## 📸 Preview

*<img width="2537" height="1282" alt="Screenshot 2026-04-04 174252" src="https://github.com/user-attachments/assets/1bdf607f-644f-47b4-904a-f7c7e4086919" />
<img width="2459" height="1192" alt="Screenshot 2026-04-04 174319" src="https://github.com/user-attachments/assets/dfaa8356-6bd5-4376-b0f1-b9451b28fef8" />
<img width="2119" height="1206" alt="Screenshot 2026-04-04 174346" src="https://github.com/user-attachments/assets/7caa2dd4-b492-4ec4-8e06-41f4b8ea1607" />
<img width="2451" height="1287" alt="Screenshot 2026-04-04 174403" src="https://github.com/user-attachments/assets/d52aaa47-cdd5-4181-bbfd-7e9ff07abe5d" />
*

---

##  Tech Stack

* **Language:** Python 3
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-learn (Logistic Regression, Extra Trees, TF-IDF)
* **NLP:** NLTK (Tokenization, Lemmatization, POS Tagging)
* **Visualization:** Plotly
* **API:** TMDb (The Movie Database)
* **Deployment:** Hugging Face Spaces

---

##  How It Works

### 🔹 1. Data Retrieval

User inputs a movie → TMDb API fetches:

* Movie metadata
* User reviews
* Recommendations

---

### 🔹 2. NLP Processing Pipeline

Each review undergoes:

* Cleaning → Remove HTML, symbols, noise
* Tokenization → Split into words
* Stopword Removal → Remove common words
* Lemmatization → Normalize words
* Vectorization → Convert text into TF-IDF features

---

### 🔹 3. Dual-Model Sentiment Analysis

* **Logistic Regression** → Fast and efficient for sparse TF-IDF data
* **Extra Trees Classifier** → Handles non-linearity and noisy text

👉 Final system compares both outputs to improve reliability

---

### 🔹 4. Visualization & Insights

* Sentiment distribution charts
* Model agreement metrics
* Positive vs negative word traits
* Interactive review table

---

##  Motivation

This project was built to bridge the gap between static ML models and real-world applications by integrating live data sources and building a scalable, interactive NLP system.

---

##  Installation & Local Setup

```bash
git clone https://github.com/Anshu-raj-co/CINEMALYZE-semantic-sentiment-analysis.git
cd CINEMALYZE-semantic-sentiment-analysis
pip install -r requirements.txt
streamlit run app.py
```

---

##  Security & Deployment

* API keys handled securely via environment variables
* Deployed on Hugging Face Spaces with optimized resource usage
* Designed for scalability and real-time interaction

---

##  Author

**Anshu Raj**

* Computer Science Engineering Student
* Specialization: Machine Learning & Data Science
* GitHub: https://github.com/Anshu-raj-co

---
