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

random_state = 747


save_path = './final_model/my_model.ckpt'
temp_embed = np.loadtxt('save_embed')
conv_model = ConvText(37, 300, temp_embed)
conv_model.restore(save_path=save_path)

test_str = ['node how to run app js',
            'how to push in master branch',
            'net load assemblies at runtime again',
            'merge csv row with a string match from a 2nd csv file',
            'database  how to properly construct database with one of its table has fields with multiple values',
            'time complexity of merge sort',
            'creating temp file in ruby',
            'install npm in ubuntu',
            'select query in sql']
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

seq_test_str_p = tokenizer.texts_to_sequences(test_str)
seq_test_str_p

seq_test_str= pad_sequences(seq_test_str_p, maxlen=37, padding='post', truncating='post')
seq_test_str

y_cnn_predict_str = conv_model.predict(seq_test_str, batch_size = 1)
y_cnn_predict_str

preds_str = y_cnn_predict_str[:9,:]>0.1
len(y_cnn_predict_str)
tags_token = np.loadtxt('tags', dtype='str')
for j in range(len(y_cnn_predict_str)):
    print('Question: ', test_str[j])
    print('Tags: ', [tags_token[j] for j in np.where(preds_str[j,:] == 1)[0].tolist()])
