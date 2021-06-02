import random
from typing import Callable

import numpy as np

from algorithm.ea.domain.genotype import Genotype
from algorithm.ea.domain.population import Population

class EvolutionaryAlgorithm:
    def __init__(self, n, generationsNR, mutationChance, populationSize, breedCuts, fitness_function : Callable, max_fit, x : np.ndarray, selectionSample=None):
        self.__n = n
        self.__max_fit = max_fit
        self.__fitness_function = fitness_function
        self.__mutationChance = float(mutationChance)
        self.__generationsNR = generationsNR
        self.__populationSize = populationSize
        self.__breedCuts = breedCuts
        self.__selectionSample = selectionSample
        self.__generationCounter = 1
        self.__generations = self.__init_population(x)

    def getBestUntilNow(self):
        return self.getLastGeneration().getFittest()

    def __init_population(self, x):
        population_list = []
        for i in range(x.shape[0]):
            population_list.append(Genotype(self.__n, self.__fitness_function, self.__max_fit, x[i].tolist()))
        return [Population(self.__n, self.__populationSize, population_list)]

    def __newGeneration(self, populationList=None):
        if populationList is not None:
            self.__generations.append(Population(self.__n, self.__populationSize, populationList))
        self.__generations.append(Population(self.__n, self.__populationSize, self.__generations[-1].getPopulation()))

    def __breed(self, firstIndividual, secondIndividual):
        firstParamList = firstIndividual.get_list_of_variables()
        secondParamList = secondIndividual.get_list_of_variables()

        son = []
        cuts = set([random.randint(0, self.__n) for i in range(self.__breedCuts)])

        current_param_value = firstParamList
        for i in range(0, self.__n):
            if i in cuts:
                current_param_value = secondParamList if current_param_value is firstParamList else firstParamList
            son.append(current_param_value[i])

        return son

    def __chooseCrossoverStyle(self):
        if self.__selectionSample is None:
            return self.__crossoverRandom()
        else:
            return self.__crossoverTournament()

    def __crossoverRandom(self):
        lastGeneration = self.__generations[-1]
        firstIndividual = random.choice(lastGeneration.getPopulation())
        secondIndividual = random.choice(lastGeneration.getPopulation())

        while firstIndividual == secondIndividual:
            secondIndividual = random.choice(lastGeneration.getPopulation())

        return self.__breed(firstIndividual, secondIndividual)

    def __crossoverTournament(self):
        if self.__selectionSample < 2:
            raise Exception("Sample too small")
        elif self.__selectionSample > self.__populationSize:
            raise Exception("Sample too big")
        samplePopulationToReproduce = sorted(
            random.sample(self.__generations[-1].getPopulation(), self.__selectionSample),
            key=lambda individual: individual.getFitness())

        return self.__breed(samplePopulationToReproduce[0], samplePopulationToReproduce[1])

    def __mutate(self, individual):
        if self.__mutationChance > random.random():
            random.shuffle(individual)
        return individual

    def __killLeastFit(self):
        lastGeneration = self.getLastGeneration().getPopulation()
        lastGeneration.remove(max(lastGeneration, key=lambda individ: individ.getFitness()))
        return lastGeneration

    def runAlgorithm(self):
        while self.__generationCounter < self.__generationsNR:
            newIndividual = Genotype(self.__n, self.__fitness_function, self.__max_fit, self.__mutate(self.__chooseCrossoverStyle()))
            newGeneration = self.__killLeastFit()
            newGeneration.append(newIndividual)
            self.__newGeneration(newGeneration)
            if not len(self.getLastGeneration().getPopulation()) is self.__populationSize:
                raise Exception("WTF")
            self.__generationCounter += 1
        return self.getLastGeneration().getFittest()

    def getN(self):
        return self.__n

    def getGenerations(self):
        return self.__generations[:]

    def getLastGeneration(self):
        return self.__generations[-1]

    def __str__(self) -> str:
        return str(self.__generations)