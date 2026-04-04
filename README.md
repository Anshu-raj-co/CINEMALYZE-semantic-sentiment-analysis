🎬 Cinemalyze: Advanced Movie Sentiment & Semantic Engine
Cinemalyze is a high-performance NLP application that goes beyond simple "Positive/Negative" labels. By combining Ensemble Machine Learning (Extra Trees Classifier) with Semantic Vector Indexing, the application provides deep insights into audience perception and thematic similarity in movie reviews.

🚀 Live Demo
https://anshuraj1212-cinemalyze.hf.space/

✨ Key Features
Dual-Engine Sentiment Analysis: Utilizes a custom-trained Extra Trees Classifier to achieve high accuracy in detecting nuanced emotions in text.
Semantic Similarity Engine: Leveraging a Semantic Index, the app identifies and retrieves existing reviews that are contextually related to the user's input, not just keyword-matched.
Deep NLP Preprocessing: Implements a rigorous pipeline including tokenization, stop-word removal, and lemmatization.
Linguistic Insights: Uses Part-of-Speech (POS) tagging to extract and display the "Top Adjectives" used in reviews to summarize the vibe of the critique.
Model Agreement Score: A unique metric that compares the prediction against semantic neighbors to ensure reliability.

🛠️ Tech Stack
Language: Python 3.13
Frontend: Streamlit (Reactive UI)
Machine Learning: Scikit-Learn (Extra Trees, TF-IDF Vectorization)
Natural Language Processing: NLTK (Punkt, WordNet, Stopwords, Perceptron Tagger)
Serialization: Joblib (for efficient loading of 185MB+ model files)
Deployment: Hugging Face Spaces (CPU Upgrade with 16GB RAM)

🧠 How It Works
1. The NLP Pipeline
Every input review undergoes a multi-stage transformation:
Cleaning: Removal of HTML tags, special characters, and numbers.
Tokenization: Breaking sentences into individual semantic units using punkt_tab.
POS Tagging: Identifying adjectives (JJ) and nouns (NN) to understand descriptive context.
Lemmatization: Reducing words to their root form (e.g., "watched" -> "watch") to normalize the input.

2. The Model Architecture
While many sentiment projects use simple Logistic Regression, Cinemalyze uses an Extra Trees (Extremely Randomized Trees) ensemble.
Why Extra Trees? It provides better generalization and reduces variance by introducing random splits in the decision trees, making it highly robust against the "noise" often found in informal movie reviews.

3. Semantic Vector Space
The project uses a pre-computed Semantic Index. By calculating the Cosine Similarity between the user's input and a massive dataset of movie reviews, the app can find "thematically identical" content even if different words are used.

📦 Installation & Local Setup
To run this project on your local machine:

Clone the Repository:
Bash
git clone https://github.com/Anshu-raj-co/CINEMALYZE-semantic-sentiment-analysis.git
cd CINEMALYZE-semantic-sentiment-analysis

Install Dependencies:
Bash
pip install -r requirements.txt

Run the Application:
Bash
streamlit run app.py
📂 Project Structure
Plaintext
├── .streamlit/          # Streamlit configuration
├── app.py               # Main application logic & UI
├── requirements.txt     # Python dependencies
├── .gitignore           # Secure exclusion of tokens and local caches
├── extra_tree_model.joblib # Trained Sentiment Model
└── semantic_index.joblib   # Vector search index for semantic lookup
🛡️ Security & Deployment Note
This project follows professional security practices by using .gitignore to prevent the leakage of Hugging Face Write Tokens. The application is hosted on an upgraded Hugging Face Space to ensure the 185MB+ models load smoothly without memory bottlenecks.

👨‍💻 Author
Anshu Raj
Role: Computer Science Engineering Student
Specialization: Machine Learning & Data Science
GitHub: @Anshu-raj-co
Project Title: Cinemalyze - Semantic Sentiment Analysis Engine
