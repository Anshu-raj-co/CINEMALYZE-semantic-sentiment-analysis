import nltk
print("Starting the download for necessary NLTK data packages...")

packages = [
    'stopwords',
    'punkt',
    'wordnet',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'averaged_perceptron_tagger_eng',
    'maxent_ne_chunker_tab' # Added the package for NER tabbing
]

for package in packages:
    print(f"Downloading '{package}'...")
    try:
        nltk.download(package)
    except Exception as e:
        print(f"Could not download {package}. Error: {e}")


print("\nAll necessary NLTK data packages have been successfully downloaded.")
print("You can now run the 'movie_sentiment_analyzer.py' script.")

