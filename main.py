from simple_perceptron import SimplePerceptron

def default_activation(value):
    if value >= 0: return 1
    return -1

input_data= [[-1,-1], [-1,1], [1,-1], [1,1]]
expected_values = [-1, -1, -1, 1]

perceptron = SimplePerceptron(input_data, expected_values, default_activation)

perceptron.train_perceptron()

print("[ -1 && -1 =", perceptron.guess([-1,-1]),
        "]\n [ -1 && 1 =" , perceptron.guess([-1, 1]),
        "]\n [ 1 && -1 =", perceptron.guess([1, -1]),
        "]\n [ 1 && 1 =", perceptron.guess([1, 1]), 
        "]"
    )