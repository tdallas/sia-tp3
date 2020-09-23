from math import e
import numpy as numpy
from simple_perceptron import SimplePerceptron
from parse import Parser

parser = Parser('input.csv', 'output.csv')

def sigmoide(value):
    return 1 / (1 + numpy.exp(-1  * value))

def de_sigmoide(value):
    return sigmoide(value) * (1 - sigmoide(value))

inputs = parser.get_inputs()
outputs = parser.get_outputs()
outputs_normalized = numpy.zeros(len(outputs))
max_value = numpy.max(outputs)
min_value = numpy.min(outputs)
i = 0
while(i < len(outputs)):
    outputs_normalized[i] = (outputs[i][0] - min_value) / (max_value - min_value)
    i += 1

perceptron = SimplePerceptron(inputs, outputs_normalized, sigmoide, de_sigmoide, eta=0.001, iterations=1000)
perceptron.train()

i = 0
while(i < len(outputs_normalized)):
    print("input: ", inputs[i])
    print("expected value: ", outputs_normalized[i])
    print("result value: ", perceptron.guess(inputs[i]))
    i += 1
