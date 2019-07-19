from django.shortcuts import render
from django.http import HttpResponse
from .ques_retrieve import *

# Create your views here.
retrieve_model = ExtractQuery('../ques-so.json')
def home(request):
    global retrieve_model
    if request.method == "POST":
        req_dict = request.POST
        sentence = req_dict['search']
        output = retrieve_model.predict(sentence)
        # print(output)
        titles = []
        all_questions = []
        for item in retrieve_model.ques:
            all_questions.append(item[0])
        for item in output:
            titles.append( {'qtitle' : item['qtitle'], 'qurl' : item['qurl']} )
        temp = {
            'all_questions': all_questions[:50],
            'output': titles,
        }
        return render(request, 'main.html', temp)        
    return render(request, 'main.html')

def profile(request):
    return render(request, 'userprofile.html')

def signin(request):
    return render(request, 'signin.html')
def signup(request):
    return render(request, 'signup.html')
def post(request):
    return render(request, 'post_question.html')
