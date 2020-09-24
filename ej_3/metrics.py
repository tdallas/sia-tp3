import matplotlib.pyplot as plt

class Metrics():
    def __init__(self, training_accuracies_array, test_accuracies_array, iterations_array):
        self.training_accuracies_array = training_accuracies_array
        self.test_accuracies_array = test_accuracies_array
        self.iterations_array = iterations_array

    def graph(self):
        plt.plot(self.iterations_array, self.training_accuracies_array, label="train")
        plt.plot(self.iterations_array, self.test_accuracies_array, label="test")
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.legend(title='Accuracy vs Epochs')
        plt.show()