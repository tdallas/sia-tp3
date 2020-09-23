import numpy as numpy
from multi_layer_perceptron import MultilayerPerceptron

#input
_input = [[-1, -1], [-1,1], [1, -1], [1, 1]]
_expected = [[-1], [1], [1], [-1]]

learning_rate = 0.7
momentum = 0.9
test_p = 0.25

beta = 0.3 #TODO: ver que valor poner

def sigmoide(value):
    return 1 / (1 + numpy.exp(-1  * value))

def de_sigmoide(value):
    return sigmoide(value) * (1 - sigmoide(value))

nn = MultilayerPerceptron(learning_rate, momentum, sigmoide, de_sigmoide, False, test_p)

nn.entry_layer(2)
nn.add_hidden_layer(2)
nn.output_layer(1)

error = nn.train(_input, _expected, epochs=10000)
#print(error)

for i in range(0, len(_input)):
    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))