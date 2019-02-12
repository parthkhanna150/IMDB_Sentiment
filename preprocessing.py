from bs4 import BeautifulSoup
import re, string, unicodedata

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def regular_preprocess(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
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
