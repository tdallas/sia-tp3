import numpy as numpy
from perceptron import Perceptron

class SimplePerceptron():
    def __init__(self, training_inputs, training_expected_values, sign_function, eta=0.01, error_delta=0.001, iterations=10000):
        self.error_delta = error_delta
        self.iterations = iterations
        self.sign_function = sign_function
        self.input_size = len(training_inputs[0])
        
        self.training_inputs = numpy.array(training_inputs)

        self.training_expected_values = training_expected_values
        self.weights = self.generate_random_weights()
        self.eta = eta
    
    def predict(self, xi, bias=0.1):
        # print('xi', xi)
        # print('weights', self.weights)
        # print(numpy.dot(self.weights, xi))
        return self.sign_function(numpy.dot(self.weights, xi))

    def generate_random_weights(self):
        return numpy.array(numpy.random.rand(self.input_size))

    def train_perceptron(self):
        # run iterations
        for current_iteration in range(self.iterations):
            # zippeo trainingInputs -> trainingExpectedValues
            error_count = 0
            for inputs, expected_value in zip(self.training_inputs, self.training_expected_values):
                prediction = self.predict(inputs) > 0.2
                # print('prediction', prediction)
                error = expected_value - prediction
                # print('error', error)
                # print('antes', self.weights)
                if error != 0 :
                    error_count+=1
                    for index, value in enumerate(inputs):
                        # print(self.eta * error * value)
                        self.weights[index] += self.eta * error * value
                    # print('despues', self.weights)
                # print('\n')
            if error_count == 0:
                print('iteracion numero', current_iteration)
                break
                

    def guess(self, inputs):
        new_inputs = inputs
        new_inputs = numpy.array(new_inputs)
        return self.sign_function(numpy.dot(self.weights, new_inputs))
    
