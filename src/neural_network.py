import numpy as np
from pprint import pprint
import random

class NeuralNetwork(object):
    def __init__(self, node_structure):
        self.node_structure = node_structure
        self.node_connections = zip(node_structure[:-1], node_structure[1:])
        self.biases = [np.random.randn(x, 1) for x in node_structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in self.node_connections]

    def feed_forward(self, activation):
        for bias_vector, weight_matrix in zip(self.biases, self.weights):
            activation = self.__calc_next_layer_activation(activation, weight_matrix, bias_vector)
        return activation

    def stochist_gradient_decent(self, training_examples, learning_rate, batch_size, epochos):
        number_of_training_examples = len(training_examples)
        for i in xrange(epochos):
            random.shuffle(training_examples)
            batchs = [
                training_examples[j:j + batch_size]
                for j in xrange(0, number_of_training_examples, batch_size)
            ]
            for batch in batchs:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                for training_input, actual_result in batch:
                    delta_nabla_b, delta_nabla_w = self.backpropagate(training_input, actual_result)

    def backpropagae(self, input, actual_result):

        return (delta_nabla_b, delta_nable_w)


    def __calc_next_layer_activation(self, inputs, weights, bias):
        value = np.dot(weights, inputs) + bias
        return self.__sigmoid(value)

    def __mean_squared_error(self, y, y_hat):
        return np.square(np.subtract(y, y_hat)).mean()

    def __sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def __sigmoid_prime(self, x):
        return self.__sigmoid(x)(1 - self.__sigmoid(x))

