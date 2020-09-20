from activation_function import ActivationFunction
from math import e

class SigmoideFunction(ActivationFunction):
    def __init__(self, name):
        super(name)

    def evaluate(self, value):
        return 1 + (e ** (- value))
