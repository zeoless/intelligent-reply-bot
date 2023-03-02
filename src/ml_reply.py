
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
import re
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

from classes_dict import *

my_words = []
my_classes = []
my_doc = []
ignore_words = ['?']

# pre-process text from classes_dict
for some_class in classes_dict:
    
    my_classes.append(some_class)
    
    for some_pattern in classes_dict[some_class]["pattern"]:
        
        temp_words = []
        
        raw_words = some_pattern
        # raw_words = ' '.join(raw_words)
        word_tokens = nltk.word_tokenize(raw_words)
        
        for some_word in word_tokens:
            if some_word not in ignore_words:
                stemmed_word = stemmer.stem(some_word.lower())
                my_words.append(stemmed_word)
                temp_words.append(stemmed_word)

    my_doc.append((temp_words, some_class))

my_words = sorted( list(set(my_words)) ) # remove duplicate words
my_classes = sorted( list(set(my_classes)) )

training = []
output = []
output_empty = [0] * len(my_classes)

for some_doc in my_doc:
    
    bag = []
    pattern_words = some_doc[0]
    
    # create bag of words array