from flask import Flask
from flask import request
from algorithm.gnSolver import GNSolver
import numpy as np

from apscheduler.schedulers.background import BackgroundScheduler


gaussNewtonAlgirithm = None

COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]
COEFFICIENTS = np.array(COEFFICIENTS)

app = Flask(__name__)

def compute_hr_mean_last_seconds():
    pass

scheduler = BackgroundScheduler()
scheduler.add_job(func=compute_hr_mean_last_seconds, trigger="interval", seconds=5)
scheduler.start()

hrReadings = []
hrReadingsMeans = []

@app.route("/test")
def testPage():
    return "Request Works"

@app.route("/hrReading", methods=['POST'])
def hrReadingFunction():
    user = request.form
    if("hrReading" in request.form.keys()):
        hrReading = user["hrReading"]
        print(hrReading)
        hrReadings.append(hrReading)
        response = app.response_class(
            response=f"Heart rate reading:{hrReading}",
            status=200
        )
    else:
        response = app.response_class(
            response=f"Missing or invalid fields!",
            status=400
        )
    return response
  
@app.route("/init")
def initGaussNewtonAlgorithm():
    gaussNewtonAlgirithm = GNSolver()

def gaussFunction(xValues : np.ndarray, coeff : list): 
    if(len(xValues.shape) != 1):
        new_array = np.ndarray(xValues.shape)
        for list_index in range(xValues.shape[0]):
            new_array[list_index] = gaussFunction(xValues[list_index], coeff)
        return new_array
    else:
        return xValues.dot(coeff)     