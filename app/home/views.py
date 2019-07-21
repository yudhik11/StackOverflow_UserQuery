from django.shortcuts import render
from django.http import HttpResponse
from .ques_retrieve import *
import json

# Create your views here.
retrieve_model = Predict()

# with open('../ans-score.json','r') as f:
#     ans_so = json.load(f)

def home(request):
    global retrieve_model
    global ans_so
    if request.method == "POST":
        req_dict = request.POST
        sentence = req_dict['search']
        ans_so, output = retrieve_model.predict(sentence.strip())
        print(ans_so)
        print(output)
        titles = []
        all_questions = []
        for item in retrieve_model.top_ques.ques:
            all_questions.append( {'qtitle':item[0], 'qurl':item[2]} )
        
        for item in output:
            ans = []
            qid = str(item['qid'])
            try:
                for ans_score in ans_so[qid]:
                    if ans_score['sentimental_score'] > 0.0:
                        sentiment = 'positive'
                    elif ans_score['sentimental_score'] == 0.0:
                        sentiment = 'neutral'
                    else:
                        sentiment = 'negative'
                    sentimental_sc = ans_score['sentimental_score']
                    sc = ans_score['score']
                    if np.isnan(sentimental_sc):
                        sentimental_sc = 0.0
                    sentimental_sc = "{:.5f}".format(sentimental_sc)
                    sc = "{:.5f}".format(sc)
                    ans.append({
                        'aid':str(ans_score['aid']),
                        'senti_score':str(sentimental_sc),
                        'upvote':str(ans_score['upvotes']),
                        'sentiment':sentiment,
                        'score':str(sc)
                    })
            except:
                pass
            titles.append( {'qtitle' : item['qtitle'], 'qurl' : item['qurl'], 'aid': ans} )
        temp = {
            'all_questions': all_questions[:50],
            'output': titles,
            'show' : True,
            'sentence' : sentence
        }
        print("="*25,"Done Fetching","="*25)
        return render(request, 'main.html', temp)
    else:        
        return render(request, 'main.html')

def profile(request):
    return render(request, 'userprofile.html')

def signin(request):
    return render(request, 'signin.html')
def signup(request):
    return render(request, 'signup.html')
def post(request):
    return render(request, 'post_question.html')
