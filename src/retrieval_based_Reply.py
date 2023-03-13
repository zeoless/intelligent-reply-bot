

import nltk
import random
import string
import sys
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from classes_dict import *

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
