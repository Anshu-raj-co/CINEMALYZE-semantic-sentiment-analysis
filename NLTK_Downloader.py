import nltk
import os

# 1. Create a local folder for the data so Render 'saves' it
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

print(f"⏳ Starting build-time download to: {nltk_data_dir}")

packages = [
    'stopwords', 'punkt', 'wordnet', 
    'averaged_perceptron_tagger', 'maxent_ne_chunker', 
    'words', 'averaged_perceptron_tagger_eng', 
    'maxent_ne_chunker_tab'
]

for package in packages:
    print(f"Downloading '{package}'...")
    try:
        # We force it into our local folder
        nltk.download(package, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        print(f"❌ Error downloading {package}: {e}")

print("\n✅ All NLTK resources are pre-staged for deployment.")
