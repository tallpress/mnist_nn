import cPickle
import gzip
import numpy as np

class MnistLoader(object):
    def load_data_wrapper():
        training_data, validation_data, test_data = self.__load_data()
        training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
        training_results = [self.__vectorized_result(y) for y in training_data[1]]
        training_data = zip(training_inputs, training_results)
        validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
        validation_data = zip(validation_inputs, validation_data[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data = zip(test_inputs, test_data[1])
        return (training_data, validation_data, test_data)

    def __load_data():
        f= gzip.open('../data/mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()
        return (training_data, validation_data, test_data)

    def __vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
