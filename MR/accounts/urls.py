from django.urls import path
from . import views

urlpatterns = [
    path('user/',views.UserLogin),
    path('recommender/',views.Recommendations),
]
