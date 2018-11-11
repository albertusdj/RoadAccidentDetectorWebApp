from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import os
import pickle

model = pickle.load(open(os.getcwd() + "\models" + "\multinomialNB"), "r")

# Create your views here.
def index(request):
  context = {}
  return render(request, 'accidentDetector/index.html', context)

def predict(request):
  tweet = request.get_json()['tweet']

  prediction = model.predict(tweet)
  response = {}
  response['prediction'] = prediction

  return JsonResponse(response)