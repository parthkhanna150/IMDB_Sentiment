from bs4 import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

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

def advanced_preprocess(text):
    return text

def preprocess(data, advanced=False):
    refined_data=[]

    if advanced:
        for data_point in data:
            refined_data.append(advanced_preprocess(data_point))
    else:
        for data_point in data:
            refined_data.append(regular_preprocess(data_point))

    return refined_data
