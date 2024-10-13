import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

# Configure logging
logging.basicConfig(filename='/tmp/app.log', level=logging.DEBUG)

# Initialize Flask application
application = Flask(__name__)
app = application

# Load models
try:
    with open('models/ridge.pkl', 'rb') as f:
        ridge = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)
    logging.debug('Models loaded successfully')
except Exception as e:
    logging.error(f'Failed to load models: {str(e)}')
    ridge = None
    standard_scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extract form data
            input_data = [
                float(request.form.get(field))
                for field in ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
            ]
            
            # Perform prediction
            if standard_scaler is not None and ridge is not None:
                new_data_scaled = standard_scaler.transform([input_data])
                result = ridge.predict(new_data_scaled)
                return render_template('home.html', results=result[0])
            else:
                logging.error('Models not loaded properly')
                return render_template('home.html', error="Model prediction failed")
        except Exception as e:
            logging.error(f'Prediction error: {str(e)}')
            return render_template('home.html', error="An error occurred during prediction")
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)