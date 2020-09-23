from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class MultilayerPerceptron():
    def __init__(self, hidden_layers, output_layer):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for _ in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.output_layer.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.hidden_layers[0].synaptic_weights += layer1_adjustment
            self.output_layer.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        outputs = []
        input = inputs
        print('inputs', inputs)
        for i in range(len(self.hidden_layers)):
            print(i, ':',input, '|',self.hidden_layers[i].synaptic_weights)
            input = self.__sigmoid(dot(input, self.hidden_layers[i].synaptic_weights))
            outputs.append(input)
        ## output for output layer
        print('output')
        outputs.append(self.__sigmoid(dot(input, self.output_layer.synaptic_weights)))
        return outputs

    def print_weights(self):
        print("    Hidden layers\n")
        for i in range(len(self.hidden_layers)):
            print(self.hidden_layers[i].synaptic_weights)
        print("\n    Output layer 2 (1 neuron, with 4 inputs):")
        print(self.output_layer.synaptic_weights)
