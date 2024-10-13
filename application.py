import os
import pickle
import numpy as np
from flask import Flask, request, render_template

application = Flask(__name__)

# Load the models
model_path = os.path.join(os.path.dirname(__file__), 'models', 'ridge.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

with open(model_path, 'rb') as f:
    ridge = pickle.load(f)
with open(scaler_path, 'rb') as f:
    standard_scaler = pickle.load(f)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
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
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8000)