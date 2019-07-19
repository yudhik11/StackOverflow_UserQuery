from __future__ import print_function
import json,sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch



def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

class ExtractQuery(object):
    def __init__(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.ans = []
        self.ques = []
        #module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        module_url = "/home/yudhik/tf"


        self.embed_fn = embed_useT(module_url)

        for line in data:
            self.ans.append(self.embed_fn([line['question']])[0])
            self.ques.append( (line['question'], line['qid'], line['qurl']) )
        self.ans=np.array(self.ans)
    
    def predict(self, sentence):
        embed = self.embed_fn([sentence])[0]
        scores = np.inner(embed, self.ans)
        idxs = scores.argsort()[-5:][::-1]
        output = []

        for idx in idxs:
            item = {
                'qtitle' : self.ques[idx][0],
                'qid' : self.ques[idx][1],
                'similarity_score' : str(scores[idx]),
                'qurl' : self.ques[idx][2]
            }
            output.append(item)

        return output
