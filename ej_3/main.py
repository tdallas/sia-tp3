from numpy import exp, array, random, dot
from multilayer_perceptron import MultilayerPerceptron, NeuronLayer


#Seed the random number generator
random.seed(1)

# Create layer 1 (3 neurons, each with 2 inputs)
layer1 = NeuronLayer(3, 2)

# Create layer 2 (a single neuron with 3 inputs)
layer2 = NeuronLayer(1, 3)

# Combine the layers to create a neural network
neural_network = MultilayerPerceptron(layer1, layer2)

print("Stage 1) Random starting synaptic weights: ")
neural_network.print_weights()

# The training set. We have 7 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = array([[0, 0], [0,1], [1, 0], [1, 1]])
training_set_outputs = array([[0], [1], [1], [0]])

# Train the neural network using the training set.
# Do it 60,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 6000)

print("Stage 2) New synaptic weights after training: ")
neural_network.print_weights()

# Test the neural network with a new situation.
print("Stage 3) Considering a new situation [1, 1] -> ?: ")
hidden_state, output = neural_network.think(array([0, 1]))
print(output)

hidden_state, output = neural_network.think(array([0, 0]))
print(output)

hidden_state, output = neural_network.think(array([1, 0]))
print(output)

hidden_state, output = neural_network.think(array([1, 1]))
print(output)