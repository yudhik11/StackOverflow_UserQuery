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
    def __init__(self):
        with open('ques-so.json', 'r') as f:
            data = json.load(f)
        self.ans = []
        self.ques = []
        module_url = "../../../tf"


        self.embed_fn = embed_useT(module_url)

        for line in data:
            self.ans.append(self.embed_fn([line['question']])[0])
            self.ques.append( (line['question'], line['qid']) )
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
                'similarity_score' : str(scores[idx])
            }
            output.append(item)
        with open('ans-so.json','w') as f:
            json.dump(output, f, indent=4, sort_keys=True)
        return output

c = ExtractQuery()
c.predict("node how to run node app.js")