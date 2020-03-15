import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import warnings; warnings.simplefilter('ignore')
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
import gensim
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import ConvText
import pickle
import nltk
import sys
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from difflib import SequenceMatcher
import readTags

random_state = 747

def getWord(word):
    res = []
    score = 0
    for token in tags:
        score1 = SequenceMatcher(None, word, token).ratio()
        if score1>0.72:
            score = score1
            res.append(token)
    return res

def extractKeywords(query):
    word_tokens = word_tokenize(query) # Sentence tokenized
    print("Word Tokens:", word_tokens)

    lemmatizer = WordNetLemmatizer()
    word_tokens = [lemmatizer.lemmatize(token) for token in word_tokens]

    filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words] # Word tokens after the removal of the stop words
    uniqueWords = []
    for i in filtered_sentence:
        if not i in uniqueWords:
            uniqueWords.append(i);
    print("Filter:", uniqueWords)

    # Keyword Extraction
    keywords = []
    for token in uniqueWords:
        # if token in tags:
        #     keywords += token + ';'
        try:
            keywords.append(getWord(token))
        except:
            continue

    # keywords = keywords[:-1]
    op_arr = keywords[:]
    #print('Op_arr:', op_arr)
    final=""
    for i in op_arr:
        for j in i:
            try:
                final += j + ';'
            except:
                continue
    final = final[:-1]
    print("Keywords:", final)

save_path = './final_model/my_model.ckpt'
temp_embed = np.loadtxt('save_embed')
conv_model = ConvText(37, 300, temp_embed)
conv_model.restore(save_path=save_path)

query = input().lower()
test_str = []
temp_str = re.sub(r"[^a-zA-Z0-9#+-]", " ", query.lower())
test_str.append(temp_str)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

seq_test_str_p = tokenizer.texts_to_sequences(test_str)
seq_test_str_p

seq_test_str= pad_sequences(seq_test_str_p, maxlen=37, padding='post', truncating='post')
seq_test_str

y_cnn_predict_str = conv_model.predict(seq_test_str, batch_size = 1)
y_cnn_predict_str

preds_str = y_cnn_predict_str[:1,:]>0.1
len(y_cnn_predict_str)
tags_token = np.loadtxt('tags', dtype='str')
for j in range(len(y_cnn_predict_str)):
    print('Question: ', test_str[j])
    print('Tags: ', [tags_token[j] for j in np.where(preds_str[j,:] == 1)[0].tolist()])

keywords = [tags_token[j] for j in np.where(preds_str[j,:] == 1)[0].tolist()]
print(keywords)

final =""
for i in keywords:
    try:
        final += i + ";"
    except:
        continue

final = final[:-1]
if len(keywords)==0:
    tags = readTags.readFile('tags-so.txt')

    # Stop Words Removal from the input sentence
    stop_words = set(stopwords.words('english')) # Set of the stopwords in the nltk corpus
    extractKeywords(temp_str)

print('CNN Keywords:', final)
test_str.remove(temp_str)
