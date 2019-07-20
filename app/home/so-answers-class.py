import requests
import json
import numpy as np
import ast

class ExtractAnswer:
    def __init__(self, path):
        self.id_arr = []
        self.ans_url=[]
        self.ans_id=[]
        self.ques_id=[]
        self.url_init_ans = 'https://stackoverflow.com/a/'
        self.url_init_ques = 'https://stackoverflow.com/q/'
        with open(path, 'r') as f:
            op = json.load(f)
            for it in op:
                self.id_arr.append(it['qid'])
        self.final_ids =""
        for i in self.id_arr:
            try:
                self.final_ids += str(i) + ";"
            except:
                continue
        self.final_ids = self.final_ids[:-1]

    def ans(self):
        u='https://api.stackexchange.com/2.2/questions/'+self.final_ids+'/answers'
        start = True
        i = 1
        while start:
            print(i)
            p={'page':str(i), 'pagesize':'100','fromdate':'1388534400','order':'desc','sort':'votes','site':'stackoverflow','key':'hWdB8OaWM0hGZP3sRV18iA(('}
            r = requests.get(url = u, params = p)
            data = r.json()
            temp = data['items']
            for j in range(0,len(temp)):
                if data['items'][j]["answer_id"] not in self.ans_id:
                    self.ans_id.append({'aid':data['items'][j]["answer_id"], 'qid':data['items'][j]["question_id"], 'upvotes':data['items'][j]["score"]})
                # if data['items'][j]["question_id"] not in ques_id:
                #     ques_id.append(data['items'][j]["question_id"])
                self.ans_url.append({'qid':data['items'][j]["question_id"], 'qurl':self.url_init_ques+str(data['items'][j]["question_id"]), 'aid':data['items'][j]["answer_id"], 'aurl':self.url_init_ans+str(data['items'][j]["answer_id"]), 'upvotes':data['items'][j]["score"]})
            if data['has_more'] == False:
                break
            else:
                i = i+1
        return self.ans_url, self.ans_id
