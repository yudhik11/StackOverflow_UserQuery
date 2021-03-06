from __future__ import print_function
import json,sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import ast
import random
import requests
import collections
import re
import os
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from .similar_ques import *

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

class ExtractQuery(object):
    def __init__(self):
        module_url = "home/tf"
        self.embed_fn = embed_useT(module_url)

    def predict(self, sentence, data):

        self.ans = []
        self.ques = []
        sent = []
        for line in data:
            sent.append(line['question'].strip())
            self.ques.append( (line['question'], line['qid'], line['qurl']) )
        self.ans = self.embed_fn(sent)
        self.ans=np.array(self.ans)
        print(self.ans.shape)
        embed = self.embed_fn([sentence])[0]
        scores = np.inner(embed, self.ans)
        idxs = scores.argsort()[-5:][::-1]
        output = []
        for idx in idxs:
            item = {
                'qtitle' : str(self.ques[idx][0]),
                'qid' : str(self.ques[idx][1]),
                'similarity_score' : str(scores[idx]),
                'qurl' : str(self.ques[idx][2])
            }
            output.append(item)
        keydict = dict(zip(self.ques, scores))
        self.ques.sort(key=keydict.get)
        self.ques = self.ques[::-1]
        return output

class ExtractAnswer:
    def __init__(self):
        self.ans_url=[]
        self.ans_id=[]
        self.url_init_ans = 'https://stackoverflow.com/a/'
        self.url_init_ques = 'https://stackoverflow.com/q/'
    
    def ans(self, op):
        self.ans_id = []
        self.ans_url = []
        self.id_arr = []

        self.op = op
        for it in self.op:
            self.id_arr.append(it['qid'])
        self.final_ids =""
        for i in self.id_arr:
            try:
                self.final_ids += str(i) + ";"
            except:
                continue
        self.final_ids = self.final_ids[:-1]
        print(self.final_ids)
        u='https://api.stackexchange.com/2.2/questions/'+self.final_ids+'/answers'
        start = True
        i = 1
        stackoverflow_key = os.environ['STACK_API_KEY']

        while start:
            print("Ans:",i)
            p={'page':str(i), 'pagesize':'100','fromdate':'1388534400','order':'desc','sort':'votes','site':'stackoverflow','key':stackoverflow_key}
            r = requests.get(url = u, params = p)
            data = r.json()
            temp = data['items']
            for j in range(0,len(temp)):
                if data['items'][j]["answer_id"] not in self.ans_id:
                    self.ans_id.append({'aid':data['items'][j]["answer_id"], 'qid':data['items'][j]["question_id"], 'upvotes':data['items'][j]["score"]})
                print(data['items'][j]["question_id"])
                self.ans_url.append({'qid':data['items'][j]["question_id"], 'qurl':self.url_init_ques+str(data['items'][j]["question_id"]), 'aid':data['items'][j]["answer_id"], 'aurl':self.url_init_ans+str(data['items'][j]["answer_id"]), 'upvotes':data['items'][j]["score"]})
            if data['has_more'] == False:
                break
            else:
                i = i+1
        return self.ans_url, self.ans_id

class ExtractComments:
    def __init__(self):
        self.ans_id = []
        self.comment_body = []
        self.score_arr = []
        iam_key = os.environ['IBM_WATSON_KEY']
        self.natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12',iam_apikey=iam_key,url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')
        

    def commentsAnalysis(self, op):
        self.op = op
        self.grouped = {}
        for it in self.op:
            try:
                self.grouped[it['qid']].append( [it['aid'], it['upvotes']] )
            except:
                self.grouped[it['qid']] = [ [it['aid'], it['upvotes']] ]

        self.ans_score = {}
        
        for key, value in self.grouped.items():
            for ans in value:
                u='https://api.stackexchange.com/2.2/answers/'+str(ans[0])+'/comments'
                start = True
                i = 1
                stackoverflow_key = os.environ['STACK_API_KEY']
                while start:
                    print("Comm:", i)
                    
                    p={'page':str(i), 'pagesize':'100','order':'desc','sort':'votes','site':'stackoverflow','key':stackoverflow_key, 'filter':'!-*jbN.o5AChs'}
                    r = requests.get(url = u, params = p)
                    data = r.json()
                    temp = data['items']
                    self.comment_body = []
                    for j in range(0,len(temp)):
                        self.comment_body.append(data['items'][j]["body"])
                    self.comment_body = self.comment_body[:10]
                    self.score_arr = []        
                    for item in self.comment_body:
                        print(item)
                        response = self.natural_language_understanding.analyze(text=item,features=Features(sentiment=SentimentOptions()), language='en').get_result()
                        self.score_arr.append(response['sentiment']['document']['score'])
                        # self.score_arr.append(random.random())
                    
                    if str(key) not in self.ans_score:
                        self.ans_score[str(key)] = []
                    self.ans_score[str(key)].append( {'aid':ans[0], 'sentimental_score':np.average(self.score_arr), 'upvotes':int(ans[1])} )
                    
                    if data['has_more'] == False:
                        break
                    else:
                        i = i+1
        return(self.ans_score)

class Predict(object):

    def __init__(self):
        self.simi_ques = SimilarQuestion()
        self.top_ques = ExtractQuery()
        self.ans_class = ExtractAnswer()
        self.comm_class = ExtractComments()

    def predict(self, sentence):
        tag_sim_ques = self.simi_ques.tags(sentence)
        sim_ques = self.top_ques.predict(sentence, tag_sim_ques)
        _,  ques_ans = self.ans_class.ans(sim_ques)
        # ques_ans = ques_ans[:5]
        ques_ans_comment = self.comm_class.commentsAnalysis(ques_ans)
        
        def cmp(upvote, score):
            if np.isnan(score):
                score=0.0  
            return (np.tanh(upvote / 1e4) * np.exp(5 * score))

        for key in ques_ans_comment:
            ques_ans_comment[key].sort(key=lambda x: cmp(x['upvotes'], x['sentimental_score']), reverse=True)
            print(ques_ans_comment[key])
            ques_ans_comment[key] = ques_ans_comment[key][:5]
            for i in range(len(ques_ans_comment[key])):
                ques_ans_comment[key][i]['score'] = cmp(ques_ans_comment[key][i]['upvotes'], ques_ans_comment[key][i]['sentimental_score'])
        return ques_ans_comment, sim_ques