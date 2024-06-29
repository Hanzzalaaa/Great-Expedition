import numpy as np
import pandas as pd
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class Prediction(APIView):
    def post(self, request):
        # review= request.GET.get('review')
        review = request.data.get('review')
        count_vect = ApiConfig.model2
        finalReview =count_vect.transform([review])
        reviews = ApiConfig.model
        PredictionMade = reviews.predict(finalReview)
        #predict using independent variables
        response_dict = {"prediction": PredictionMade}
        return Response(response_dict, status=200)



# city_budgets = {
#     'Hunza': 8000,
#     'Saif-ul-Mulook': 3000,
#     'Naran Kaghan': 8000,
#     'Balakot': 6000,
#     'Shogran': 5000,
#     'Arankel': 6000,
#     'Kashmir': 8000,
#     'Upper-neelum': 5000,
#     'Nathia Gali': 3000,
#     'Fairy Meadows': 7000,
#     'Deosai Plains': 5000,
#     'Skardu': 8000,
#     'Ziyarat': 2000,
#     'Quetta': 3000,
#     'Kund Malir Beach': 2000,
#     'Ormara Beach': 3000,
# }

# class TourBudget(APIView):
#     def post(self, request):
#         budget = float(request.data.get('budget'))
#         direction = request.data.get('direction')
        
#         if budget <= 6000:
#             return Response({"Message":"Your budget is too low"},status = 400)
 
#         # Load the dataset
#         data = pd.read_csv('city_data.csv')  # Replace 'city_data.csv' with your dataset file

#         # Preprocess the data
#         label_encoder_city = LabelEncoder()
#         label_encoder_direction = LabelEncoder()

#         data['City'] = label_encoder_city.fit_transform(data['City'])
#         data['Direction'] = label_encoder_direction.fit_transform(data['Direction'])

#         # Split the data into features (X) and target (y)
#         X = data[['Budget', 'Direction']]
#         y = data['City']

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train the machine learning model
#         model = RandomForestClassifier()
#         model.fit(X_train, y_train)

#         # Load the LabelEncoders
#         label_encoder_city = ApiConfig.model3
#         label_encoder_direction = ApiConfig.model5

#         # Encode the user inputs
#         if direction in label_encoder_direction.classes_:
#             direction_input = label_encoder_direction.transform([direction])
#         else:
#             raise ValueError(f"Unseen direction label: {direction}")

#         # Predict the recommended cities based on budget
#         remaining_budget = budget - 6000
#         recommended_cities = []

#         while remaining_budget > 0:
#             city_encoded = model.predict([[remaining_budget, direction_input[0]]])
#             recommended_city = label_encoder_city.inverse_transform(city_encoded)
            
#             city_budget = city_budgets.get(recommended_city[0])
            
#             if city_budget and city_budget <= remaining_budget:
#                 recommended_cities.append(recommended_city[0])
#                 remaining_budget -= city_budget
                
#                 # Remove the selected city from the available cities
#                 data = data[data['City'] != city_encoded[0]]
#                 X = data[['Budget', 'Direction']]
#                 y = data['City']
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                 model.fit(X_train, y_train)
#             else:
#                 break


#         print(recommended_cities)
#         response_dict = {"recommended_cities": recommended_cities}
#         return Response(response_dict, status=200)

#         # Output the result
#     print("Recommended cities based on your budget and direction:")





class TourBudget(APIView):
    def post(self, request):
        budget = float(request.data.get('budget'))
        direction = request.data.get('direction')
        
        if budget <= 6000:
            return Response({"Message": "Your budget is too low"}, status=400)
        
        # Load the dataset
        data = pd.read_csv('city_data.csv')  # Replace 'city_data.csv' with your dataset file

        # Preprocess the data
        label_encoder_city = LabelEncoder()
        label_encoder_direction = LabelEncoder()

        data['City'] = label_encoder_city.fit_transform(data['City'])
        data['Direction'] = label_encoder_direction.fit_transform(data['Direction'])

        # Split the data into features (X) and target (y)
        X = data[['Budget', 'Direction']]
        y = data['City']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the machine learning model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Save the LabelEncoders
        joblib.dump(label_encoder_city, 'label_encoder_city.joblib')
        joblib.dump(label_encoder_direction, 'label_encoder_direction.joblib')

        # Load the LabelEncoders
        label_encoder_city = joblib.load('label_encoder_city.joblib')
        label_encoder_direction = joblib.load('label_encoder_direction.joblib')

        # Encode the user inputs
        if direction in label_encoder_direction.classes_:
            direction_input = label_encoder_direction.transform([direction])
        else:
            raise ValueError(f"Unseen direction label: {direction}")

        # Predict the recommended cities based on budget
        remaining_budget = budget
        recommended_cities = []

        while remaining_budget > 0:
            # Filter data based on direction
            filtered_data = data[data['Direction'] == direction_input[0]]

            if len(filtered_data) == 0:
                break

            # Randomly select a city from the filtered data
            selected_city = filtered_data.sample(n=1)

            city_encoded = selected_city['City'].values[0]
            recommended_city = label_encoder_city.inverse_transform([city_encoded])[0]
            city_budget = selected_city['Budget'].values[0]

            if city_budget <= remaining_budget:
                recommended_cities.append(recommended_city)
                remaining_budget -= city_budget

                # Remove the selected city from the available cities
                data = data[data['City'] != city_encoded]

                # Re-split the data into training and testing sets
                X = data[['Budget', 'Direction']]
                y = data['City']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the machine learning model
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
            else:
                break

        response_dict = {"recommended_cities": recommended_cities}
        return Response(response_dict, status=200)


