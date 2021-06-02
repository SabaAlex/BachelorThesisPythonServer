class Population:
    def __init__(self, n, populationSize, population):
        self.__n = n
        self.__populationSize = populationSize

        self.__population = population[:]

    def getMeanFitness(self):
        return sum(map(lambda individual: individual.getFitness(), self.__population)) / self.__populationSize


    def getPopulation(self):
        return self.__population[:]

    def getFittest(self):
        return min(self.__population, key=lambda individ: individ.getFitness())

    def addIndividual(self, individual):
        if len(self.__population) < self.__populationSize:
            self.__population.append(individual)
        else:
            raise Exception("Cannot insert a new member in a population at the max size")

    def removeIndividual(self, individual):
        self.__population.remove(individual)

    def __str__(self) -> str:
        separator = '\n '
        return separator.join([str(x) for x in self.__population])