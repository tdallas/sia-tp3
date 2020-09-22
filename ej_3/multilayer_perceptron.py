import numpy as numpy

class MultilayerPerceptron():
    def __init__(self, training_inputs, training_expected_values, activation_function, de_activation_function, eta=0.25, iterations=10):
        self.eta = eta
        self.iterations = iterations
        self.input_size = len(training_inputs[0])
        self.training_inputs = numpy.array(training_inputs)
        self.training_expected_values = training_expected_values
        self.weights = self.generate_random_weights()
        self.activation_function = activation_function
        self.de_activation_function = de_activation_function
    
    def generate_random_weights(self):
        return numpy.array(numpy.random.rand(self.input_size + 1))

    def predict(self, input):
    
    def calculate_square_error(self, expected_value, prediction):
        return 0.5 * ((expected_value - prediction) ** 2)

    def train(self):

    def guess(self, input):
        