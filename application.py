import logging
logging.basicConfig(filename='/tmp/application.log', level=logging.DEBUG)

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

logging.debug(f"NumPy version: {np.__version__}")
logging.debug(f"Pandas version: {pd.__version__}")
logging.debug(f"Scikit-learn version: {StandardScaler().__class__.__module__}")

application = Flask(__name__)
app = application

# Load the models
try:
    ridge = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    logging.debug("Models loaded successfully")
except Exception as e:
    logging.error(f"Failed to load models: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge.predict(new_data_scaled)
            logging.debug(f"Prediction result: {result[0]}")
            return render_template('home.html', results=result[0])
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return render_template('home.html', error="An error occurred during prediction")
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)