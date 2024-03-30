from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Create Flask app
app = Flask(__name__, template_folder='template', static_folder='template/static')

# Load the pickle model
with open("model.pkl", "rb") as file:
    combined_models = pickle.load(file)

# Extract individual models and scaler
model_rf = combined_models['RandomForestRegressor']
model_xgb = combined_models['XGBoostRegressor']
scaler = combined_models['StandardScaler']

# Load the original dataframe to get dummy variable columns
df = pd.read_csv("C:/House Price Project/data_reduce.csv")

def preprocess_input(bath, balcony, total_sqft_int, bhk, availability, location):
    # Set the current month dynamically
    current_month = datetime.now().strftime('%B')  # Use '%B' for full month name

    # Pass availability options to the template
    availability_options = ["Ready To Move", "This Month"]

    # Exclude the current month from the options
    all_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    availability_options += [month for month in all_months if month != current_month]

    # Create a new dataframe with the input data
    input_data = pd.DataFrame({
        'bath': [bath],
        'balcony': [balcony],
        'total_sqft_int': [total_sqft_int],
        'bhk': [bhk],
        'availability': [availability],
        'location': [location]
    })

    # Convert categorical columns to dummy variables
    input_data = pd.get_dummies(input_data, columns=['availability', 'location'])

    # Align the input columns with the original dataframe to ensure consistency
    input_data = input_data.reindex(columns=df.columns, fill_value=0)

    # Use only the columns used during fitting the scaler
    features_used = df.columns.drop(['price'])  # Exclude the target variable
    x_scaled = scaler.transform(input_data[features_used])

    return x_scaled, availability_options, current_month

@app.route('/')
def Home():
    _, availability_options, current_month = preprocess_input(0, 0, 0, 0, "Ready To Move", "Kathmandu")
    return render_template('index.html', current_month=current_month, availability_options=availability_options)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        bath = float(request.form['bath'])
        balcony = float(request.form['balcony'])
        total_sqft_int = float(request.form['total_sqft_int'])
        bhk = int(request.form['bhk'])
        availability = request.form['availability']
        location = request.form['location']

        # Validation checks
        if total_sqft_int < 120:
            _, availability_options, current_month = preprocess_input(bath, balcony, total_sqft_int, bhk, availability, location)
            return render_template('index.html', prediction_text='Total Square Foot must be greater than or equal to 120.', availability_options=availability_options,
                                   bath=bath, balcony=balcony, total_sqft_int=total_sqft_int, bhk=bhk, availability=availability, location=location)

        if bhk <= 0:
            _, availability_options, current_month = preprocess_input(bath, balcony, total_sqft_int, bhk, availability, location)
            return render_template('index.html', prediction_text='BHK value must be greater than 0.', availability_options=availability_options,
                                   bath=bath, balcony=balcony, total_sqft_int=total_sqft_int, bhk=bhk, availability=availability, location=location)
        # Preprocess the input using the loaded scaler
        x_scaled, availability_options, _ = preprocess_input(bath, balcony, total_sqft_int, bhk, availability, location)

        # Debugging print statement
        print("Availability Options:", availability_options)

        # Predictions from each model
        prediction_rf = model_rf.predict(x_scaled)[0]
        prediction_xgb = model_xgb.predict(x_scaled)[0]

        # Combine predictions (averaging)
        combined_prediction = (prediction_rf + prediction_xgb) / 2

        # Multiply by 100000 if needed
        prediction = combined_prediction * 100000

        return render_template('index.html', prediction_text='Predicted Price is Rs.{}'.format(prediction),
                               availability_options=availability_options,
                               bath=bath, balcony=balcony, total_sqft_int=total_sqft_int, bhk=bhk,
                               availability=availability, location=location)

    # If not a POST request (i.e., manual page reload), clear form values
    _, availability_options, current_month = preprocess_input(0, 0, 0, 0, "Ready To Move", "Kathmandu")
    return render_template('index.html', prediction_text='', availability_options=availability_options, bath='', balcony='', total_sqft_int='', bhk='', availability='', location='')

if __name__ == '__main__':
    app.run(debug=True)