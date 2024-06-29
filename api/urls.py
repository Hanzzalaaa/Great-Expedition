from django.urls import path
from .views import *

urlpatterns = [
    path('', Prediction.as_view(), name = 'prediction'),
    path('tour/', TourBudget.as_view(), name = 'TourBudget'),
]