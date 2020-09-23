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
            outputs = self.think(training_set_inputs)

            # Calculate the error for all layers
            # errors[0] is error from output layer
            errors, deltas = self.get_errors(training_set_outputs, outputs)

            print('errors', errors)
            print('deltas', deltas)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.hidden_layers[0].synaptic_weights += layer1_adjustment
            self.output_layer.synaptic_weights += layer2_adjustment

#             # Calculate the error for layer 1 (By looking at the weights in layer 1,
#             # we can determine by how much layer 1 contributed to the error in layer 2).
#             layer1_error = layer2_delta.dot(self.output_layer.synaptic_weights.T)
#             layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)
    def get_errors(self, training_expected_outputs, outputs):
        errors = []
        deltas = []

        # from output
        output_error = training_expected_outputs - outputs[-1]
        errors.append(output_error)
        output_delta = output_error * self.__sigmoid_derivative(outputs[-1])
        deltas.append(output_delta)
        
        #hidden layers
        previous_error = output_error
        previous_delta = output_delta
        print('pesos iniciales', self.output_layer.synaptic_weights.T)
        previous_weights = self.output_layer.synaptic_weights.T
        current_hidden_layer = len(self.hidden_layers) - 1
        index = len(outputs) - 2
        while index >= 1:
            print('adentro', index, previous_delta, previous_weights)
            previous_error = previous_delta.dot(previous_weights)
            print(previous_error)
            errors.append(previous_error)
            previous_delta = previous_error * self.__sigmoid_derivative(outputs[index])
            deltas.append(previous_delta)
            
            previous_weights = self.hidden_layers[current_hidden_layer]
            current_hidden_layer -= 1

            index -= 1
            
        return errors, deltas

    # The neural network thinks
    def think(self, inputs):
        outputs = []
        input = inputs
        # input * first hidden layer
        input = self.__sigmoid(dot(input, self.hidden_layers[0].synaptic_weights))
        outputs.append(input)
        
        # product between hidden layers
        for i in range(1, len(self.hidden_layers)):
            input = self.__sigmoid(dot(input, self.hidden_layers[i].synaptic_weights))
            outputs.append(input)

        # last product
        outputs.append(self.__sigmoid(dot(input, self.output_layer.synaptic_weights)))
        
        # print(outputs)
        # outputs size = #hiddenlayers + 1 
        return outputs

    def print_weights(self):
        print('')
        # print("    Hidden layers\n")
        # for i in range(len(self.hidden_layers)):
        #     print(self.hidden_layers[i].synaptic_weights)
        # print("\n    Output layer 2 (1 neuron, with 4 inputs):")
        # print(self.output_layer.synaptic_weights)
