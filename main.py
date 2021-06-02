from algorithm.ea.evolutionaryAlgorithm import EvolutionaryAlgorithm
import numpy as np
from algorithm.regression.gnSolver import GNSolver

COEFFICIENTS = [-0.001, 0.1, 0.1, 2.0, 15.0]

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

x = np.ones((10, 5))

y = gaussFunction(x, COEFFICIENTS)
yn = y + 3 * np.random.randn(len(x))

print(x)
print(yn)

init_guess = 1000000 * np.random.random(len(COEFFICIENTS))

solver = GNSolver(gaussFunction, init_guess, x, yn, 10 ** (-6))

print(solver.predict(np.array([1, 2, 1, 4, 1])))

solver.fitNext()
solver.fitNext()
solver.fitNext()
solver.fitNext()
solver.fitNext()
solver.fitNext()
solver.fitNext()

print(solver.predict(np.array([1, 2, 1, 4, 1])))

coeff = solver.get_coefficients()

x.shape[1]
ea_algorithm = EvolutionaryAlgorithm(x.shape[1], 500, 0.2, x.shape[0], 2, gauss_function, 95, x, 4)

fittest = ea_algorithm.runAlgorithm()
print(fittest.get_list_of_variables())