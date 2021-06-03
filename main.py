from random import random
from algorithm.ea.evolutionaryAlgorithm import EvolutionaryAlgorithm
import numpy as np
from algorithm.regression.gnSolver import GNSolver

COEFFICIENTS = [24, 51, 82, 41, 10]

def gaussFunction(xValues : np.ndarray, coeff : list) -> np.ndarray: 
    if(len(xValues.shape) != 1):
        new_array = []
        for list_index in range(xValues.shape[0]):
            new_array.append(gaussFunction(xValues[list_index], coeff))
        return new_array
    else:
        return xValues.dot(coeff)   

def gauss_function(x : list):
    sum_result = 0
    for i in range(len(x)):
        sum_result += coeff[i] * x[i]
    return sum_result

xval = []
for i in range(10):
    yval = []
    for j in range(5):
        yval.append(random())
    xval.append(yval)

x = np.array(xval)

y = gaussFunction(x, COEFFICIENTS)
yn = y + 3 * np.random.randn(len(x))

print(x)
print(yn)

init_guess = 100000 * np.random.random(len(COEFFICIENTS))

solver = GNSolver(gaussFunction, init_guess, x, yn, 10 ** (-6))

print(solver.predict(np.array([0.45, 0.12, 0.87, 0.4, 0.213])))

i = 0
isFit, new_coeff = solver.fitNext()
while not isFit:
    i += 1
    print(i)
    isFit, new_coeff = solver.fitNext()

print(solver.predict(np.array([0.45, 0.12, 0.87, 0.4, 0.213])))

coeff = solver.get_coefficients()

x.shape[1]
ea_algorithm = EvolutionaryAlgorithm(x.shape[1], 500, 0.2, x.shape[0], 2, gauss_function, 95, x, 4)

fittest = ea_algorithm.runAlgorithm()
print(fittest.get_list_of_variables())
print(solver.predict(np.array(fittest.get_list_of_variables())))
