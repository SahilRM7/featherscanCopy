import io
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.conf import settings
from django.http import JsonResponse
import base64
from .models import BirdSpecies
from .models import FAQ

def faq_view(request):
    faqs = FAQ.objects.all()
    return render(request, 'faq.html', {'faqs': faqs})

def login_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        print(user,"sample")
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return redirect('login')
    return render(request,'login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

def signup_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        User.objects.create_user(username=username, password=password)
        return redirect('/login')
    return render(request,'signup.html')

def home(request):
    faqs = FAQ.objects.all()[:4]
    return render(request,'home.html', {'faqs': faqs, 'MEDIA_URL': settings.MEDIA_URL})

def explore(request):
    birds = BirdSpecies.objects.all()
    return render(request, 'explore.html', {'birds': birds})
