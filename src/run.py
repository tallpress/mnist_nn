import numpy as np
import mnist_loader
from pprint import pprint

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# size of network
sizes = [784,30, 10]

# generate the biases, gausian distributed
initial_biases = [np.random.randn(y, 1) for y in sizes[1:]]
# print(initial_biases)

# create a matrix of all the weights, that is for each of the connections, gausian distrubuted
node_connections = zip(sizes[:-1], sizes[1:])
initial_weights = [np.random.randn(y, x) for x, y in node_connections]


def calculate_node_value(inputs, weights, bias):
    value = np.dot(weights, inputs) + bias
    return sigmoid(value)

# given an input x, what is the output of the network?
def feed_forward(x):
    for bias, weight in zip(initial_biases, initial_weights):
        x = calculate_node_value(x, weight, bias)
    return x

def mean_squared_error(y, y_hat):
    return np.square(np.subtract(y, y_hat)).mean()


# gets the mean squared error of one input...
example = mnist_loader.load_data_wrapper()[0][0]
data = example[0]
actual_result = feed_forward(data)
desired_result = example[1]
print mean_squared_error(desired_result, actual_result)

# neet to repeat and adjusting bias and weights by taking into account the error

