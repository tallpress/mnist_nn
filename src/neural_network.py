import numpy as np
from pprint import pprint
import random

class NeuralNetwork(object):
    def __init__(self, node_structure):
        self.node_structure = node_structure
        self.node_connections = zip(node_structure[:-1], node_structure[1:])
        self.biases = [np.random.randn(x, 1) for x in node_structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in self.node_connections]

    def stochist_gradient_decent(self, training_examples, learning_rate, batch_size, epochos, test_data=None):
        if test_data: n_test = len(test_data)
        number_of_training_examples = len(training_examples)
        for i in xrange(epochos):
            random.shuffle(training_examples)
            batchs = [
                training_examples[j:j + batch_size]
                for j in xrange(0, number_of_training_examples, batch_size)
            ]
            for batch in batchs:
                self.update_batch(batch, learning_rate)
                if test_data:
                    print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
                else:
                    print "Epoch {0} complete".format(j)


    def update_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(biases.shape) for biases in self.biases]
        nabla_w = [np.zeros(weights.shape) for weights in self.weights]
        for training_input, actual_result in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(training_input, actual_result)
            nabla_b = [nabla_b + delta_nabla_b for nabla_b, delta_nabla_b in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nabla_w + delta_nabla_w for nabla_w, delta_nabla_w in zip(nabla_w, delta_nabla_w)]
        self.weights = [weight - (learning_rate/len(batch)) * nabla_w for weight, nabla_w in zip(self.weights, nabla_w)]
        self.biases = [bias - (learning_rate/len(batch)) * nabla_b for bias, nabla_b in zip(self.biases, nabla_b)]

    def backpropagate(self, input, actual_result):
        activation = input
        activations = [input]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_vectors.append(z)
            activation = self.__sigmoid(z)
            activations.append(activation)

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta = self.__cost_derivative(activations[-1], actual_result) * self.__sigmoid_prime(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        number_layers = len(self.node_structure)
        for layer in xrange(2, number_layers):
            z = z_vectors[-layer]
            sigmoid_prime = self.__sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def __sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def __sigmoid_prime(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def __cost_derivative(self, output_activations, desired_result):
        return (output_activations - desired_result)

    def feed_forward(self, activation):
        for bias_vector, weight_matrix in zip(self.biases, self.weights):
            activation = self.__calc_next_layer_activation(activation, weight_matrix, bias_vector)
        return activation

    def __calc_next_layer_activation(self, inputs, weights, bias):
        value = np.dot(weights, inputs) + bias
        return self.__sigmoid(value)

    def __mean_squared_error(self, y, y_hat):
        return np.square(np.subtract(y, y_hat)).mean()

