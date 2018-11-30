class Solution(object):
    """
    A solution for the given problem, it is composed of a binary value and its fitness value
    """
    def __init__(self, value):
        self.value = value
        self.fitness = 0

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.value)