import numpy as np
from pprint import pprint
import random

class NeuralNetwork(object):
    def __init__(self, node_structure):
        self.node_structure = node_structure
        self.node_connections = zip(node_structure[:-1], node_structure[1:])
        self.initial_biases = [np.random.randn(x, 1) for x in node_structure[1:]]
        self.initial_weights = [np.random.randn(y, x) for x, y in self.node_connections]

    def feed_forward(self, activation):
        for bias_vector, weight_matrix in zip(self.initial_biases, self.initial_weights):
            activation = self.__calc_next_layer_activation(activation, weight_matrix, bias_vector)
        return activation

    def __calc_next_layer_activation(self, inputs, weights, bias):
        value = np.dot(weights, inputs) + bias
        return self.__sigmoid(value)

    def __sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def __sigmoid_prime(self, x):
        return self.__sigmoid(x)(1 - self.__sigmoid(x))

    def __mean_squared_error(self, y, y_hat):
        return np.square(np.subtract(y, y_hat)).mean()

    def output_error(self, )


