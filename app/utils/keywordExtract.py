import nltk
import numpy as np
import sys
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from difflib import SequenceMatcher

import readTags

tags = readTags.readFile('tags-so.txt')

# Stop Words Removal from the input sentence
stop_words = set(stopwords.words('english')) # Set of the stopwords in the nltk corpus

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

query = input()
query = query.lower()
extractKeywords(query)
