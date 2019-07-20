from django.urls import path
from django.conf.urls import include, url
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home.html/', views.home, name='home'),
    path('userprofile.html/', views.profile, name='profile'),
    path('post.html/', views.post, name='post'),
    path('signin.html/', views.signin, name='signin'),
    path('signup.html/', views.signup, name='signup'),
]
