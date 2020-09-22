from simple_perceptron import SimplePerceptron
import numpy as numpy

print('AND: ')
# AND input data
input_data = numpy.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])
# AND expected result for input
input_expected_data = numpy.array([-1, -1, -1, 1])

perceptron = SimplePerceptron(input_data, input_expected_data)
perceptron.train()
print(perceptron.guess([-1,-1]))
print(perceptron.guess([-1,1]))
print(perceptron.guess([1,-1]))
print(perceptron.guess([1,1]))

print('OR: ')

# OR input data
input_data = numpy.array([
        [-1, 1],
        [1, -1],
        [-1, -1],
        [1, 1]
    ])
# OR expected result for input
input_expected_data = numpy.array([1, 1, -1, -1])

perceptron = SimplePerceptron(input_data, input_expected_data)
perceptron.train()
print(perceptron.guess([-1,1]))
print(perceptron.guess([1,-1]))
print(perceptron.guess([-1,-1]))
print(perceptron.guess([1,1]))