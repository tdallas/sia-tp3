from math import e
import numpy as numpy
from simple_perceptron import SimplePerceptron
from parser import Parser

parser = Parser('ej_2/input.csv', 'ej_2/output.csv')

def sigmoide(value):
    return 1 / (1 + (e ** (- value)))

def de_sigmoide(value):
    return (e ** (- value)) / ((1 + (e**-value)) ** 2)

perceptron = SimplePerceptron(parser.get_inputs(), parser.get_outputs(), sigmoide, de_sigmoide)
perceptron.train()
print(perceptron.guess([4.4793,-4.0765,4.4558]))
print("expected value", 87.3174)

# print('OR: ')
# # OR input data
# or_input_data = numpy.array([
#         [-1, 1],
#         [1, -1],
#         [-1, -1],
#         [1, 1]
#     ])
# # OR expected result for input
# or_input_expected_data = numpy.array([1, 1, -1, 1])

# or_perceptron = SimplePerceptron(or_input_data, or_input_expected_data)
# or_perceptron.train()
# print(or_perceptron.guess([-1,1]))
# print(or_perceptron.guess([1,-1]))
# print(or_perceptron.guess([-1,-1]))
# print(or_perceptron.guess([1,1]))


# print('XOR: ')
# # OR input data
# xor_input_data = numpy.array([
#         [-1, 1],
#         [1, -1],
#         [-1, -1],
#         [1, 1]
#     ])
# # OR expected result for input
# xor_input_expected_data = numpy.array([1, 1, -1, -1])

# xor_perceptron = SimplePerceptron(xor_input_data, xor_input_expected_data)
# xor_perceptron.train()
# print(xor_perceptron.guess([-1,1]))
# print(xor_perceptron.guess([1,-1]))
# print(xor_perceptron.guess([-1,-1]))
# print(xor_perceptron.guess([1,1]))