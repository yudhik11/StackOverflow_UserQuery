from __future__ import print_function
import json,sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch

#module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

module_url = "/home/yudhik/tf"

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


embed_fn = embed_useT(module_url)

with open('ques-so.txt', 'r') as f:
    data = f.readlines()
ans = []
ques = []
cnt=0
for line in data:
    cnt+=1
    print(cnt)
    line = line.strip()
    line = eval(line)
    ans.append(embed_fn([line['question']])[0])
    ques.append( (line['question'], line['qurl']) )
ans=np.array(ans)

query = "node how to run node app.js"
embed = embed_fn([query])[0]

scores = np.inner(embed, ans)
idxs= scores.argsort()[-5:][::-1]

output = []

for idx in idxs:
    item = {
        'qtitle' : ques[idx][0],
        'qurl' : ques[idx][1],
        'similarity_score' : scores[idx]
    }
    output.append(item)
    print(ques[idx], scores[idx])

with open('ans-so.json','w') as f:
    json.dump(output, f, indent=4, sort_keys=True)
