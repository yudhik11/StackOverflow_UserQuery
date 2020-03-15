from __future__ import print_function
import json,sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

#module_url = "/home/yudhik/tf"

def embed_useT(module)
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
for line in data:
    ans.append(embed_fn([line.strip()])[0])
ans=np.array(ans)

query = "node how to run node app.js"
embed = embed_fn([query])[0]

scores = np.inner(embed, ans)
idxs= scores.argsort()[-5:][::-1]
for idx in idxs:
    print(data[idx], scores[idx])

