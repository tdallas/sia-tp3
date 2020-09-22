import numpy as numpy
from perceptron import Perceptron

class SimplePerceptron():
    
    def __init__(self, training_inputs, training_expected_values, eta=0.1, iterations=1000):
        self.eta = eta
        self.iterations = iterations
        self.input_size = len(training_inputs[0])
        self.training_inputs = numpy.array(training_inputs)
        self.training_expected_values = training_expected_values
        self.weights = self.generate_random_weights()
    
    def generate_random_weights(self):
        return numpy.array(numpy.random.rand(self.input_size + 1))

    def activation_function(self, x):
        if x >= 0: return 1
        return -1
 
    def predict(self, input):
        dot_product = self.weights.T.dot(input)
        return self.activation_function(dot_product)
 
    def train(self):
        for _ in range(self.iterations):
            for input, expected_value in zip(self.training_inputs, self.training_expected_values):
                input_with_bias = numpy.insert(input, 0, 1)
                prediction = self.predict(input_with_bias)
                error = expected_value - prediction
                self.weights = self.weights + self.eta * error * input_with_bias

    def guess(self, input):
        return self.predict(numpy.insert(input, 0, 1))
    
