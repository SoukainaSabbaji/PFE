from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def UserLogin(request):
   return render(request, 'accounts/login.html')

def Recommendations(request):
    HttpResponse('recommender')
