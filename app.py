import functools
from random import randint
import json as json_package
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

import logging

logging.basicConfig(filename='myapp.log', level=logging.INFO)

logger = logging.getLogger(__name__)

connection_counter = FastReadCounter()

gaussNewtonAlgirithm = None

vo2MAX = 56.7

hrMin = 43

hrReading_error = 0.05

xValues = []
yValues = []
hrReadings = []

SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/server.db'
socketio = SocketIO(app)
db = SQLAlchemy(app)
db.create_all()
migrate = Migrate(app, db)

class CoefficientData(db.Model):
    __tablename__ = 'coefficients'
    id = db.Column(db.Integer, primary_key=True)
    coefficients = db.Column(db.String(120))
    rating = db.Column(db.String(120))

def compute_vo2_current(current_reading, error, hr_min, vo2_max):
    return (((current_reading * (1 - error)) / hr_min * 15.3) / vo2_max) * 100

def compute_hr_mean_last_seconds():
    try:
        emit("get parameters", {}, broadcast=True)
        average_reading = functools.reduce(lambda a, b: (a * 0.75) + (b * 0.25), hrReadings)
        yValues.append(compute_vo2_current(average_reading, hrReading_error, hrMin, vo2MAX))
        hrReadings.clear()
    except Exception:
        pass


scheduler = BackgroundScheduler()
scheduler.add_job(func=compute_hr_mean_last_seconds, trigger="interval", seconds = 5)

def load_data_from_db():
    try:
        result = CoefficientData.query.all()

        DB_COEFFICIENTS = []
        RATINGS = []

        for row in result:
            DB_COEFFICIENTS.append(map(lambda element: int(element), str(row[1]).split(',')))
            RATINGS.append(int(row[2]))

        coeff_size = len(DB_COEFFICIENTS)
        DB_COEFFICIENTS = np.array(DB_COEFFICIENTS)

        ACCUMULATOR = DB_COEFFICIENTS[0] * RATINGS[0]
        for i in range(1, len(coeff_size)):
            ACCUMULATOR = ACCUMULATOR + DB_COEFFICIENTS[i] * RATINGS[i]
            
        return ACCUMULATOR / sum(RATINGS)
    except Exception:
        return None

#routes
@app.route("/test")
def testPage():
    logger.info("Loggint test run")
    return "Request Works"

@app.route("/hrReading", methods=['POST'])
def hrReadingFunction():
    user = request.form
    if("hrReading" in request.form.keys()):
        hrReading = user["hrReading"]

        if connection_counter.get_count() != 0:
            hrReadings.append(int(hrReading))

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

    db_loaded = load_data_from_db()

    np_x_Values = np.array(xValues)

    gauss_newton_coeff = [randint(5, 50) for i in range(np_x_Values.shape[1])]

    if db_loaded is not None:
        gauss_newton_coeff = db_loaded

    gauss_newton_coeff = np.array(gauss_newton_coeff)

    gauss_newton_algirithm = GNSolver(gaussFunction, gauss_newton_coeff, np.array(xValues), np.array(yValues), tolerance_difference=5)

    isFit, new_coeff = gauss_newton_algirithm.fitNext()
    while not isFit:
        isFit, new_coeff = gauss_newton_algirithm.fitNext()

    newData = CoefficientData(','.join(new_coeff), str(randint(1, 5)))
    db.session.add(newData)
    db.session.commit()

    emit("configured parameters", {"data" : new_coeff}, broadcast=True)
    

def gaussFunction(xValues : np.ndarray, coeff : list): 
    if(len(xValues.shape) != 1):
        new_array = np.ndarray(xValues.shape)
        for list_index in range(xValues.shape[0]):
            new_array[list_index] = gaussFunction(xValues[list_index], coeff)
        return new_array
    else:
        return xValues.dot(coeff)     


#socket app
@socketio.on("get max variables")
def get_max_variables(json):

    data = json_package.loads(json)
    values_list = data['data']
    coeff = list(map(lambda element: int(element), values_list))

    def gauss_function(x : list):
        sum_result = 0
        for i in range(len(x)):
            sum_result += coeff[i] * x[i]
        return sum_result

    np_x_Values = np.array(xValues)

    ea_algorithm = EvolutionaryAlgorithm(np_x_Values.shape[1], 500, 0.2, np_x_Values.shape[0], 2, gauss_function, 95, np_x_Values, 4)

    fittest = ea_algorithm.runAlgorithm().get_list_of_variables()

    emit("set max variables", {"data" : fittest})

@socketio.on("get min variables")
def get_min_variabless(json):

    data = json_package.loads(json)
    values_list = data['data']
    coeff = list(map(lambda element: int(element), values_list))

    def gauss_function(x : list):
        sum_result = 0
        for i in range(len(x)):
            sum_result += coeff[i] * x[i]
        return sum_result

    np_x_Values = np.array(xValues)

    ea_algorithm = EvolutionaryAlgorithm(np_x_Values.shape[1], 500, 0.2, np_x_Values.shape[0], 2, gauss_function, 20, np_x_Values, 4)

    fittest = ea_algorithm.runAlgorithm().get_list_of_variables()

    emit("set min variables", {"data" : fittest})

@socketio.on("add parameters")
def add_parameters(json):
    data = json_package.loads(json)
    values_list = data['data']
    int_values_list = list(map(lambda element: int(element), values_list))
    xValues.append(int_values_list)

@socketio.on('connect')
def test_connect():
    connection_counter.increment()
    emit('connection response', {'data': 'Connected'})
    logger.info('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    connection_counter.decrement()
    logger.info('Client disconnected')
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, "0.0.0.0", 8080)