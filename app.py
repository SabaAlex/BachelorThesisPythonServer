from algorithm.ea.evolutionaryAlgorithm import EvolutionaryAlgorithm
from flask import Flask
from flask import request
from flask_socketio import emit, SocketIO
from algorithm.regression.gnSolver import GNSolver
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from apscheduler.schedulers.background import BackgroundScheduler
from utils.atomic_counter import FastReadCounter

connection_counter = FastReadCounter()

gaussNewtonAlgirithm = None

app = Flask(__name__)
socketio = SocketIO(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)


def compute_hr_mean_last_seconds():
    socketio.emit("get parameters", namespace="/param")
    try:
        result = sum(hrReadings) / len(hrReadings)
        yValues.append(result)
        hrReadings.clear()
    except Exception:
        pass

scheduler = BackgroundScheduler()
scheduler.add_job(func=compute_hr_mean_last_seconds, trigger="interval", seconds=5)

xValues = []
yValues = []

hrReadings = []

#routes
@app.route("/test")
def testPage():
    return "Request Works"

@app.route("/hrReading", methods=['POST'])
def hrReadingFunction():
    user = request.form
    if("hrReading" in request.form.keys()):
        hrReading = user["hrReading"]

        if connection_counter.get_count() != 0:
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

@app.route("/start_calibration", methods=['GET'])
def start_calibration():
    scheduler.start()
    return app.response_class(
            response=f"Started Calibration",
            status=200
        )

@app.route("/end_calibration", methods=['GET'])
def end_calibration():
    scheduler.shutdown()
    return app.response_class(
            response=f"Ended Calibration",
            status=200
        )

@app.route("/init")
def init_gauss_newton_algorithm():

    gauss_newton_coeff = [100, 0.1, 0.1, 2, 15]
    gauss_newton_coeff = np.array(gauss_newton_coeff)

    gauss_newton_algirithm = GNSolver(gaussFunction, gauss_newton_coeff, xValues, yValues, tolerance_difference=5)

    isFit, new_coeff = gauss_newton_algirithm.fitNext()
    while not isFit:
        isFit, new_coeff = gauss_newton_algirithm.fitNext()

    emit("configured parameters", {"data" : new_coeff}, boradcast=True)
    

def gaussFunction(xValues : np.ndarray, coeff : list): 
    if(len(xValues.shape) != 1):
        new_array = np.ndarray(xValues.shape)
        for list_index in range(xValues.shape[0]):
            new_array[list_index] = gaussFunction(xValues[list_index], coeff)
        return new_array
    else:
        return xValues.dot(coeff)     

#socket app
@socketio.on("max variables", "/var")
def get_max_variables(coeff):

    def gauss_function(x : list):
        sum_result = 0
        for i in range(len(x)):
            sum_result += coeff[i] * x[i]
        return sum_result

    ea_algorithm = EvolutionaryAlgorithm(xValues.shape[1], 500, 0.2, xValues.shape[0], 2, gauss_function, 95, xValues, 4)

    fittest = ea_algorithm.runAlgorithm().get_list_of_variables()

    emit("max variablex", {"data" : fittest})

@socketio.on("min variables", "/var")
def get_min_variabless(coeff):

    def gauss_function(x : list):
        sum_result = 0
        for i in range(len(x)):
            sum_result += coeff[i] * x[i]
        return sum_result

    ea_algorithm = EvolutionaryAlgorithm(xValues.shape[1], 500, 0.2, xValues.shape[0], 2, gauss_function, 20, xValues, 4)

    fittest = ea_algorithm.runAlgorithm().get_list_of_variables()

    emit("max variablex", {"data" : fittest})

@socketio.on("get parameters","/param")
def get_parameters(json):
    xValues.append(json)

@socketio.on('connect')
def test_connect():
    connection_counter.increment()
    emit('connection response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    connection_counter.decrement()
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, "0.0.0.0", 8080)