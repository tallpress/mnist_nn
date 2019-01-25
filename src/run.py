from mnist_loader import MnistLoader
from neural_network import NeuralNetwork

loader = MnistLoader()
training_data, validation_data, test_data = loader.load_data_wrapper()

nn = NeuralNetwork([784, 20, 10])
nn.stochist_gradient_decent(training_data, 0.5, 5, 2, test_data)

