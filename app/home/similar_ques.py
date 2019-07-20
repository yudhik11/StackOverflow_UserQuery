import numpy as np
import pandas as pd
import time
import json
import requests
import re
import warnings; warnings.simplefilter('ignore')
# from sklearn.utils import shuffle
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_sample_weight
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.naive_bayes import GaussianNB
# import gensim
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.model import ConvText
import pickle
import nltk
import sys
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from difflib import SequenceMatcher
from utils import readTags

random_state = 747

TAGS = readTags.readFile('utils/tags-so.txt')

def getWord(word):
    res = []
    score = 0
    for token in TAGS:
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
    stop_words = set(stopwords.words('english'))
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
    op_arr = keywords[:]
    final=""
    for i in op_arr:
        for j in i:
            try:
                final += j + ';'
            except:
                continue
    final = final[:-1]
    return final

class SimilarQuestion:
    def __init__(self):
        # self.query = query
        self.root = 'utils/'
        save_path = self.root+'final_model/my_model.ckpt'
        self.temp_embed = np.loadtxt(self.root+'save_embed')
        self.conv_model = ConvText(37, 300, self.temp_embed)
        self.conv_model.restore(save_path=save_path)

        self.temp_str = ""#re.sub(r"[^a-zA-Z0-9#+-]", " ", query.lower())
        with open(self.root+'tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        # self.seq_test_str_p = tokenizer.texts_to_sequences(self.test_str)
        # self.seq_test_str= pad_sequences(self.seq_test_str_p, maxlen=37, padding='post', truncating='post')

    def tags(self, query):
        self.test_str = []
        self.query = query
        self.temp_str = re.sub(r"[^a-zA-Z0-9#+-]", " ", self.query.lower())
        self.test_str.append(self.temp_str)
        seq_test_str_p = self.tokenizer.texts_to_sequences(self.test_str)
        seq_test_str= pad_sequences(seq_test_str_p, maxlen=37, padding='post', truncating='post')
        y_cnn_predict_str = self.conv_model.predict(seq_test_str, batch_size = 1)
        preds_str = y_cnn_predict_str[:1,:]>0.1
        tags_token = np.loadtxt(self.root+'tags', dtype='str')
        for j in range(len(y_cnn_predict_str)):
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
            # Stop Words Removal from the input sentence
            stop_words = set(stopwords.words('english')) # Set of the stopwords in the nltk corpus
            extractKeywords(self.temp_str)
        print('CNN Keywords:', final)
        self.test_str.remove(self.temp_str)
        self.query = self.query.lower()
        u='https://api.stackexchange.com/2.2/similar'
        tag_arr=[]
        start = True
        i = 1
        while start:
            print(i)
            p={'page':str(i), 'pagesize':'100','fromdate':'1388534400','tagged':final,'title':self.query, 'order':'desc','sort':'votes','min':'40','site':'stackoverflow','key':'hWdB8OaWM0hGZP3sRV18iA(('}
            r = requests.get(url = u, params = p)
            data = r.json()
            temp = data['items']
            for j in range(0,len(temp)):
                tag_arr.append({'question':data['items'][j]['title'], 'qid':data['items'][j]["question_id"], 'qurl':data['items'][j]["link"]})
            if data['has_more'] == False:
                break
            else:
                i = i+1
        return tag_arr
