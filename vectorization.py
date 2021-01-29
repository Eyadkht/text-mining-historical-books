"""
Created on Sat Mar  7 21:12:26 2020

@author: eyadk
"""
import pickle
import string
import pandas as pd
from __future__ import print_function
import nltk, sklearn
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#################### Loading Files ##########################
documents = open("document_dictionary.pkl","rb")
vocabulary = open("total_vocab.pkl","rb")
doc_text = open("document_text.pkl","rb")

documents_dic = pickle.load(documents)
total_vocab = pickle.load(vocabulary)
doc_text = pickle.load(doc_text)
vocabulary.close()
documents.close()

##################### Tokenization ############################
porter_stemmer=nltk.stem.porter.PorterStemmer()
lemmatizer = WordNetLemmatizer()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

list_stopwords = nltk.corpus.stopwords.words('english')
list_stopwords.extend( [ ':', ';', '[', ']', '"', "'", '(', ')', '.', '?','#','!',
                        ',','@']
                      )

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def tokenizer_stemmer(s):
    s = s.lower()

    tokens = nltk.tokenize.word_tokenize( text = s )        # Split strings into words
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    cleaned = [word for word in stripped if word.isalpha()]
    stop_words = [t for t in cleaned if t not in list_stopwords  ] # Remove stop words
    #stemmed_words = [stemmer.stem(t) for t in stop_words]   # Put words into base form
    short_words = [t for t in stop_words if len(t) > 2]              # Remove short words!
    no_ascii_digits_words = [strip_non_ascii(t) for t in short_words ]     # Remove non-ascii and digits
    no_white_space = [t.strip() for t in no_ascii_digits_words ] # Remove white space
    no_empty = [t for t in no_white_space if len(t) > 1 ]

    tokens = no_empty
    return tokens

########################### Vectorization ##########################
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=0.2, max_df=0.8,
                                 stop_words='english',
                                 ngram_range=(1,3),
                                 use_idf=True, tokenizer=tokenizer_stemmer)



tfidf_matrix = tfidf_vectorizer.fit_transform(doc_text.values()) #fit the vectorizer to synopses

f = open("tfidf_vectorizer.pkl","wb")
pickle.dump(tfidf_vectorizer,f)
f.close()

f = open("tfidf_matrix.pkl","wb")
pickle.dump(tfidf_matrix,f)
f.close()