import requests
import json
import collections
import numpy as np
import re
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

class ExtractComments:
    def __init__(self, path):
        self.ans_id = []
        self.ques_id = []
        self.final_dict = {}
        self.comment_url = []
        self.comment_body = []
        self.score_arr = []
        self.upvote_sc = 0
        self.ans_score = {}
        self.grouped = collections.defaultdict(list)
        self.natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12',iam_apikey='4pVB2V6iRJsLdJMTOuMc4ylEog_ZRe_wfopWf9-tdqc2',url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')
        with open(path, 'r') as f:
            self.op = json.load(f)
            for it in self.op:
                self.grouped[it['qid']].append(it['aid'])

        for key, val in self.grouped.items():
            self.final_dict[key]=val

    def commentsAnalysis(self):
        for key, value in self.final_dict.items():
            for ans in value:
                u='https://api.stackexchange.com/2.2/answers/'+str(ans)+'/comments'
                start = True
                i = 1
                while start:
                    print(i)
                    p={'page':str(i), 'pagesize':'100','order':'desc','sort':'votes','site':'stackoverflow','key':'hWdB8OaWM0hGZP3sRV18iA((', 'filter':'!-*jbN.o5AChs'}
                    r = requests.get(url = u, params = p)
                    data = r.json()
                    temp = data['items']
                    for j in range(0,len(temp)):
                        self.comment_url.append({'co-body':data['items'][j]["body"], 'co-url':data['items'][j]["link"]})
                        self.comment_body.append(data['items'][j]["body"])
                    for item in self.comment_body:
                        response = self.natural_language_understanding.analyze(text=item,features=Features(sentiment=SentimentOptions())).get_result()
                        self.score_arr.append(response['sentiment']['document']['score'])
                    try:
                        for item in self.op:
                            if item['aid'] == ans:
                                self.upvote_sc = item['upvotes']
                        self.ans_score.setdefault(key, [])
                        self.ans_score[key].append({'aid':ans, 'sentimental_score':np.average(self.score_arr), 'upvotes':self.upvote_sc})
                    except:
                        return 'Numpy issue or API issue\n'
                    if data['has_more'] == False:
                        break
                    else:
                        i = i+1
        return(self.ans_score)
