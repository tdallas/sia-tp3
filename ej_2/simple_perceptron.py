import numpy as numpy

class SimplePerceptron():
    
    def __init__(self, training_inputs, training_expected_values, activation_function, de_activation_function, eta=0.1, iterations=1000):
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
 
    def train(self, test_inputs, test_expected_values, delta=0.001, print_data=False):
        test_accuracies = []
        training_accuracies = []
        iters = []
        training_accuracy = 0
        # running till iterations
        for it in range(self.iterations):
            training_correct_cases = 0
            # zip training inputs with corresponding expected values
            for input, expected_value in zip(self.training_inputs, self.training_expected_values):
                # add bias=1 to input
                input_with_bias = numpy.insert(input, 0, 1)
                # calculate prediction
                prediction = self.predict(input_with_bias)
                # calculate error
                error = expected_value - prediction
                if(abs(error) < delta):
                    training_correct_cases += 1
                # update weights
                self.weights = self.weights + (self.eta * error * self.de_activation_function(self.weights.T.dot(input_with_bias)) * input_with_bias.T) 

            training_accuracies.append(training_correct_cases/len(self.training_expected_values))
            test_correct_cases = 0
            i = 0
            while (i < len(test_expected_values)):
                if(abs(test_expected_values[i] - self.guess(test_inputs[i])) < delta):
                    test_correct_cases += 1
                i += 1
            test_accuracy = test_correct_cases / len(test_expected_values)
            test_accuracies.append(test_accuracy)
            iters.append(it)
            if(print_data):
                print("Epoch: ", it)
                print("Training accuracy: ", training_accuracy)
                print("Test accuracy: ", test_accuracy)
        print("weights: ", self.weights)
        return training_accuracies, test_accuracies, iters

    def guess(self, input):
        # add bias to input (for dot product stuff)
        return self.predict(numpy.insert(input, 0, 1))
    
