import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

def cleaning(text):
    text_clean = str(text)
    text_clean = text_clean.replace('\r',' ')
    text_clean = text_clean.replace('\n',' ')
    text_clean = text_clean.replace('&amp',' ')
    text_clean = text_clean.replace('&gt',' ')
    text_clean = text_clean.replace('&lt',' ')
    text_clean = text_clean.replace('[^a-zA-Z]+',' ')
    return text_clean

def case_folding(text):
    text_cf = text
    text_cf = text_cf.lower()
    return text_cf

def lemmatization(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    text_split = text.split(" ")
    lemma_list = []
    for i in text_split:
        lemma_text = wordnet_lemmatizer.lemmatize(i, pos="v")
        lemma_list.append(lemma_text)
    lemmatized = ' '.join(map(str,lemma_list))
    return lemmatized

def stopword_removal(text):
    stopword_list = list(stopwords.words('english'))
    text_stopword = text
    text_stopword = ' '.join([i for i in text_stopword.split() if i not in stopword_list])
    return text_stopword

def preprocessing(text):
    text_preprocess = text
    text_preprocess = cleaning(text_preprocess)
    text_preprocess = case_folding(text_preprocess)
    text_preprocess = lemmatization(text_preprocess)
    text_preprocess = stopword_removal(text_preprocess)
    return text_preprocess

def tokenize(list_text):
    vects = pickle.load(open('../Data/tokenize.pkl', 'rb'))
    encoded_docs = vects.texts_to_sequences(list_text)
    padded_docs = sequence.pad_sequences(encoded_docs,maxlen=200,padding='post')
    return padded_docs