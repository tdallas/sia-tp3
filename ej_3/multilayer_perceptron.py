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
        # print('weights1', self.hidden_layers[0].synaptic_weights)
        # print('weights2', self.output_layer.synaptic_weights)
        for _ in range(number_of_training_iterations):
            # Pass the training set through our neural network
            outputs = self.think(training_set_inputs)

            # Calculate the error for all layers
            # errors[0] is error from output layer
            errors, deltas = self.get_errors(training_set_outputs, outputs)

            # print('errors', errors)
            # print('deltas', deltas)

            # Calculate how much to adjust the weights by
            adjustments = self.get_adjustments(training_set_inputs, deltas, outputs)
            # print('adjustments',adjustments)
            # print('\nhidden 1', self.hidden_layers[1].synaptic_weights)
            
            self.adjust_weights(adjustments)

    def adjust_weights(self, adjustments):
        index = 0
        while index < len(self.hidden_layers):
            # print('index', index)
            # print('pesos', self.hidden_layers[index].synaptic_weights)
            # print('adjustment[index',adjustments[index])
            self.hidden_layers[index].synaptic_weights += adjustments[index]
            index+=1

    def get_adjustments(self, training_set_inputs, deltas, outputs):
        adjustments = []
        adjustments.append(training_set_inputs.T.dot(deltas[-1]))
        
        outputs_index = len(outputs) - 2
        deltas_index = len(deltas) - 2

        while outputs_index >= 0:
            adjustments.append(outputs[outputs_index].T.dot(deltas[deltas_index]))
            outputs_index-=1
            deltas_index-=1
        return adjustments

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
        # print('pesos iniciales', self.output_layer.synaptic_weights.T)
        previous_weights = self.output_layer.synaptic_weights.T
        current_hidden_layer = len(self.hidden_layers) - 1
        index = len(outputs) - 2
        while index >= 0:
            previous_error = previous_delta.dot(previous_weights)
            errors.append(previous_error)
            previous_delta = previous_error * self.__sigmoid_derivative(outputs[index])
            deltas.append(previous_delta)
            
            previous_weights = self.hidden_layers[current_hidden_layer].synaptic_weights.T
            current_hidden_layer -= 1

            index -= 1
            
        return errors, deltas

    # The neural network thinks
    def think(self, inputs):
        outputs = []
        input = inputs
        # input * first hidden layer
        print(input)
        input = self.__sigmoid(dot(input, self.hidden_layers[0].synaptic_weights))
        outputs.append(input)
        
        i = 1
        # product between hidden layers
        while i < len(self.hidden_layers):
            input = self.__sigmoid(dot(input, self.hidden_layers[i].synaptic_weights))
            outputs.append(input)
            i+=1
        # last product
        # print('input:', input)
        # print('output_layer', self.output_layer.synaptic_weights)
        outputs.append(self.__sigmoid(dot(input, self.output_layer.synaptic_weights)))
        
        # print(outputs)
        # outputs size = #hiddenlayers + 1 
        # print('outputs',outputs)
        return outputs

    def print_weights(self):
        print('')
        # print("    Hidden layers\n")
        # for i in range(len(self.hidden_layers)):
        #     print(self.hidden_layers[i].synaptic_weights)
        # print("\n    Output layer 2 (1 neuron, with 4 inputs):")
        # print(self.output_layer.synaptic_weights)
