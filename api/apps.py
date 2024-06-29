import os
import joblib
from django.apps import AppConfig
from django.conf import settings


class ApiConfig(AppConfig):
    name = 'api'
    MODEL_FILE = os.path.join(settings.MODELS, "review.joblib")
    model = joblib.load(MODEL_FILE)
    
    MODEL2_FILE = os.path.join(settings.MODELS, "count_vect.joblib")
    model2 = joblib.load(MODEL2_FILE)
    
    # MODEL4_FILE = os.path.join(settings.MODELS, "label_encoder.joblib")
    # model4 = joblib.load(MODEL4_FILE)
    
    MODEL3_FILE = os.path.join(settings.MODELS, "label_encoder_city.joblib")
    model3 = joblib.load(MODEL3_FILE)
    
    MODEL5_FILE = os.path.join(settings.MODELS, "label_encoder_direction.joblib")
    model5 = joblib.load(MODEL5_FILE)
    
    MODEL6_FILE = os.path.join(settings.MODELS, "tour_model.joblib")
    model6 = joblib.load(MODEL6_FILE)