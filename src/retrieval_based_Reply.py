

import nltk
import random
import string
import sys
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from classes_dict import *

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_tokens(tokens):

    lemmer = nltk.stem.WordNetLemmatizer()

    return [lemmer.lemmatize(token) for token in tokens]

def lem_normalize(text):

    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    # return LemTokens(nltk.word_tokenize(text.lower()))

def my_response(my_dict, user_input, sent_tokens):

    robo_response = ''

    #sent_tokens.append(user_response)
    sent_tokens['user'] = user_input

    sent_tokens_ = []

    for value in sent_tokens:
        sent_tokens_.append(sent_tokens[value])

    # tfidf_vec = TfidfVectorizer(tokenizer = lem_normalize, stop_words='english')
    tfidf_vec = TfidfVectorizer(tokenizer = lem_normalize)

    tfidf = tfidf_vec.fit_transform(sent_tokens_)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]