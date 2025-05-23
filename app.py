from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("wine_quality_lr_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature inputs from form
        features = [float(request.form[key]) for key in request.form if key != 'wine_type']
        wine_type = request.form['wine_type']
        wine_type_encoded = 1 if wine_type == 'white' else 0
        features.append(wine_type_encoded)

        # Make prediction
        prediction = model.predict([features])[0]
        return render_template('index.html', prediction_text=f'Predicted wine quality: {round(prediction, 2)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
