import numpy as numpy

class SimplePerceptron():
    
    def __init__(self, training_inputs, training_expected_values, activation_function, de_activation_function, eta=1, iterations=3):
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
        dot_product = self.weights.T.dot(input)
        return self.activation_function(dot_product)

    def calculate_square_error(self, expected_value, prediction):
        return 0.5 * ((expected_value - prediction) ** 2)
 
    def train(self):
        # running till iterations
        for _ in range(self.iterations):
            # zip training inputs with corresponding expected values
            for input, expected_value in zip(self.training_inputs, self.training_expected_values):
                # add bias=1 to input
                input_with_bias = numpy.insert(input, 0, 1)
                # calculate prediction
                prediction = self.predict(input_with_bias)
                # calculate error
                error = expected_value - prediction
                # update weights
                self.weights = self.weights + (self.eta * error * self.de_activation_function(prediction) * input_with_bias) 
        print(self.weights)

    def guess(self, input):
        # add bias to input (for dot product stuff)
        return self.predict(numpy.insert(input, 0, 1))
    
