import random
from typing import Callable

class Genotype:
    def __init__(self, n, fitness_function : Callable, max_fit, list_of_variables=None):
        self.__n = n

        self.__list_of_variables = [x for x in list_of_variables]

        self.__max_fit = max_fit
        self.__fitness_function = fitness_function
        self.__fitness = self.__fitnessComputation()

    def getBestFitness(self):
        return self.__fitness

    def getFitness(self):
        return self.__fitness

    def __fitnessComputation(self):
        return abs((self.__max_fit * 1.0) - self.__fitness_function(self.__list_of_variables))

    def get_list_of_variables(self):
        return self.__list_of_variables[:]

    def __str__(self) -> str:
        return str(self.__list_of_variables)

    def __eq__(self, other) -> bool:
        if type(other) is list:
            return other == self.__list_of_variables
        return False