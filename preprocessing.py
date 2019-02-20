import re, string, unicodedata
# import contractions
# import inflect
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer, WordNetLemmatizer

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_links_characters(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub('\[[^]]*\]', '', text)
    return text

def regular_preprocess(text):
    text = remove_html(text)
    text = remove_links_characters(text)
    return text

def remove_stopwords(docs, stopwords):
    docs_ref = []
    for doc in docs:
        word_list = doc.lower().split()
        word_list_ref = [word for word in word_list if word not in stopwords]
        word_str_ref = ' '.join(word_list_ref)
        docs_ref.append(word_str_ref)
    return docs_ref

def stem_words(docs):
    stemmer = PorterStemmer()
    stems = []
    for doc in docs:
        word_list = doc.lower().split()
        for word in word_list:
            stem = stemmer.stem(word)
            stems.append(stem)
        stems_str = ' '.join(stems)
        stems.append(stems_str)
    return stems_str

def preprocess(data):
    refined_data = []
    for dp in data:
        refined_data.append(regular_preprocess(dp))
    return refined_data
