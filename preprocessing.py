import re, string, unicodedata
# import nltk
# nltk.download('stopwords')
# import contractions
# import inflect
from bs4 import BeautifulSoup
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer

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
    # text = stem_words(text.split())
    # text = replace_contractions(text)
    return text

# def advanced_preprocess(text):
# #     words = nltk.word_tokenize(text)
#     words = replace_numbers(text)
#     words = remove_non_ascii(words)
#     words = to_lowercase(words)
#     words = remove_stopwords(words)
#     words = stem_words(words)
#     return words

# def replace_contractions(text):
#     """Replace contractions in string of text"""
#     return contractions.fix(text)
#
# def stem_words(words):
#     """Stem words in list of tokenized words"""
#     stemmer = PorterStemmer()
#     stems = []
#     for word in words:
#         stem = stemmer.stem(word)
#         stems.append(stem)
#     return stems
#
# def remove_stopwords(words):
#     """Remove stop words from list of tokenized words"""
#     new_words = []
#     for word in words:
#         if word not in stopwords.words('english'):
#             new_words.append(word)
#     return new_words
#
# def replace_numbers(words):
#     """Replace all interger occurrences in list of tokenized words with textual representation"""
#     p = inflect.engine()
#     new_words = []
#     for word in words:
#         if word.isdigit():
#             new_word = p.number_to_words(word)
#             new_words.append(new_word)
#         else:
#             new_words.append(word)
#     return new_words
#
# def lemmatize_verbs(words):
#     """Lemmatize verbs in list of tokenized words"""
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas
#
# def to_lowercase(words):
#     """Convert all characters to lowercase from list of tokenized words"""
#     new_words = []
#     for word in words:
#         new_word = word.lower()
#         new_words.append(new_word)
#     return new_words
#
# def remove_non_ascii(words):
#     """Remove non-ASCII characters from list of tokenized words"""
#     new_words = []
#     for word in words:
#         new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#         new_words.append(new_word)
#     return new_words

def preprocess(data):
    refined_data = []
    for dp in data:
        refined_data.append(regular_preprocess(dp))

    # refined_data = advanced_preprocess(refined_data)
    return refined_data
