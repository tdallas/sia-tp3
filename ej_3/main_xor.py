from numpy import exp, array, random, dot
from multilayer_perceptron import MultilayerPerceptron, NeuronLayer

#Seed the random number generator
random.seed(1)

hidden_layer_1 = NeuronLayer(5, 2)
hidden_layer_2 = NeuronLayer(5, 5)
hidden_layer_3 = NeuronLayer(5, 5)
hidden_layer_4 = NeuronLayer(5, 5)
hidden_layer_5 = NeuronLayer(5, 5)

# Create layer 2 (a single neuron with 4 inputs)
layer2 = NeuronLayer(1, 5)

# Combine the layers to create a neural network
neural_network = MultilayerPerceptron([hidden_layer_1, hidden_layer_2, hidden_layer_3, hidden_layer_4, hidden_layer_5], layer2)

# print("Stage 1) Random starting synaptic weights: ")a

# The training set. We have 7 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = array([[0, 0], [0,1], [1, 0], [1, 1]])
training_set_outputs = array([[0], [1], [1], [0]])

# Train the neural network using the training set.
# Do it 60,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 100000)

print("Stage 2) New synaptic weights after training: ")
# neural_network.print_weights()

# Test the neural network with a new situation.
# print("Stage 3) Considering a new situation [1, 1] -> ?: ")
output = neural_network.think(array([0, 1]))
# print('todo el output', output)
print('[0 xor 1] is ~',output[-1])
print('\n')

output = neural_network.think(array([0, 0]))
# print('todo el output', output)
print('[0 xor 0] is ~',output[-1])
print('\n')

output = neural_network.think(array([1, 0]))
# print('todo el output', output)
print('[1 xor 0] is ~',output[-1])
print('\n')

output = neural_network.think(array([1, 1]))
# print('todo el output', output)
print('[1 xor 1] is ~', output[-1])
print('\n')