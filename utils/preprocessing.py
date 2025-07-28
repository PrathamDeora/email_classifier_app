from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().split()
    return " ".join([word for word in text if word not in STOPWORDS])
