from .ngrams_generator import NGramsGenerator

def download_pkgs():
    import nltk
    nltk.download('stopwords', download_dir="/tmp/nltk_data")
    nltk.download('punkt', download_dir="/tmp/nltk_data")