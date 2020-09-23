from math import e
import numpy as numpy
from simple_perceptron import SimplePerceptron
from parse import Parser

parser = Parser('input.csv', 'output.csv')

def sigmoide(value):
    return 1 / (1 + numpy.exp(-1  * value))

def de_sigmoide(value):
    return sigmoide(value) * (1 - sigmoide(value))

perceptron = SimplePerceptron(parser.get_inputs(), parser.get_outputs(), sigmoide, de_sigmoide)
perceptron.train()

print(perceptron.guess([4.4793,-4.0765,4.4558]))
print("expected value", 87.3174)

print(perceptron.guess([-4.1793,-4.9218,1.7664]))
print('expected', 1.5257)