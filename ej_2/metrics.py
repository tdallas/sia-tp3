from math import e
import numpy as numpy
from simple_perceptron import SimplePerceptron
from parse import Parser
import matplotlib.pyplot as plt

parser = Parser('ej_2/input.csv', 'ej_2/output.csv')

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

split_i = 150
train_inputs = inputs[:split_i]
train_outputs = outputs_normalized[:split_i]
test_inputs = inputs[split_i:]
test_outputs = outputs_normalized[split_i:]

perceptron = SimplePerceptron(train_inputs, train_outputs, sigmoide, de_sigmoide, eta=0.1, iterations=300)
training_accuracies, test_accuracies, iters = perceptron.train(test_inputs, test_outputs, delta=0.001, print_data=True)

plt.plot(iters, training_accuracies, label="train")
plt.plot(iters, test_accuracies, label="test")

plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(title='Accuracy vs Epochs')
plt.show()