from simple_perceptron import SimplePerceptron
import numpy as numpy

print('\n')
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
i = 0
while(i < len(input_expected_data)):
    print("input: ", input_data[i])
    print("expected value: ", input_expected_data[i])
    print("result value: ", perceptron.guess(input_data[i]))
    i += 1

print('\n')
print('OR: ')
# OR input data
or_input_data = numpy.array([
        [-1, 1],
        [1, -1],
        [-1, -1],
        [1, 1]
    ])
# OR expected result for input
or_input_expected_data = numpy.array([1, 1, -1, 1])

or_perceptron = SimplePerceptron(or_input_data, or_input_expected_data)
or_perceptron.train()
i = 0
while(i < len(or_input_expected_data)):
    print("input: ", or_input_data[i])
    print("expected value: ", or_input_expected_data[i])
    print("result value: ", or_perceptron.guess(or_input_data[i]))
    i += 1

print('\n')
print('XOR: ')
# OR input data
xor_input_data = numpy.array([
        [-1, 1],
        [1, -1],
        [-1, -1],
        [1, 1]
    ])
# OR expected result for input
xor_input_expected_data = numpy.array([1, 1, -1, -1])

xor_perceptron = SimplePerceptron(xor_input_data, xor_input_expected_data)
xor_perceptron.train()
i = 0
while(i < len(xor_input_expected_data)):
    print("input: ", xor_input_data[i])
    print("expected value: ", xor_input_expected_data[i])
    print("result value: ", xor_perceptron.guess(xor_input_data[i]))
    i += 1