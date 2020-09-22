from simple_perceptron import SimplePerceptron
from parser import Parser
import numpy as numpy

parser = Parser('ej_2/input.csv', 'ej_2/output.csv')

print(parser.get_outputs())
print(parser.get_inputs())

# print('AND: ')
# # AND input data
# input_data = numpy.array([
#         [-1, -1],
#         [-1, 1],
#         [1, -1],
#         [1, 1]
#     ])
# # AND expected result for input
# input_expected_data = numpy.array([-1, -1, -1, 1])

# perceptron = SimplePerceptron(input_data, input_expected_data)
# perceptron.train()
# print(perceptron.guess([-1,-1]))
# print(perceptron.guess([-1,1]))
# print(perceptron.guess([1,-1]))
# print(perceptron.guess([1,1]))

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