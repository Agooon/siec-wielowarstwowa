from numpy.lib.function_base import bartlett
from NeuralNetwork import NeuralNetwork
from keras.datasets import mnist
import numpy as np
import copy

(x_train, y_train), (x_text, y_test)   = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_text = x_text.reshape((x_text.shape[0], 28 * 28))

x_train = (x_train/255).astype('float64')
x_text = (x_text/255).astype('float64')

def main():
    print('Hello World')

if __name__ == "__main__":
    train_size = 6000
    test_size = 8000
    epochs = 50

    # print('Normal normal')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='normal')

    # print('Normal Xavier')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='xavier')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='normal')

    # print('Normal He')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='he')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='normal')
    # # print('-------------')

    # print()
    # print()
    # print()

    
    

    # print('normal')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    # params = copy.deepcopy(nn3.params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='normal')
    # print()
    
    # print('momentum')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=0.8)
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='momentum')
    # print()
    
    # print('momentum_nesterov')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=0.8)
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='momentum_nesterov')
    # print()
    
    # print('adagrad')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='adagrad')
    # print()

    # print('adadelta ')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='adelta')
    # print()
    
    # print('adam')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=0.8, )
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='adam')
    # print()


    print('Sigmoid')
    # print('normal')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='sig', learning_rate_mode='normal')
    # print()
    
    print('xavier')
    nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=0.8, weight_scale_type='xavier')
    nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='sig', learning_rate_mode='normal')
    print()
    
    print('he')
    nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=0.8, weight_scale_type='he')
    nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='sig', learning_rate_mode='normal')
    print()


    print('ReLU')
    print('normal')
    nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='normal')
    print()
    
    print('xavier')
    nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=0.8, weight_scale_type='xavier')
    nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='normal')
    print()
    
    print('he')
    nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=0.8, weight_scale_type='he')
    nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size], activation_function='relu', learning_rate_mode='normal')
    print()


    # print('Momentum Xavier')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='xavier')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum')

    # print('Momentum He')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='he')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum')

    # print()
    # print()
    # print()

    # print('momentum_nesterov normal')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum_nesterov')

    # print('momentum_nesterov Xavier')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='xavier')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum_nesterov')

    # print('momentum_nesterov He')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='he')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum_nesterov')

    # print()
    # print()
    # print()

    # print('adagrad normal')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='adagrad')

    # print('adagrad Xavier')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='xavier')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='adagrad')

    # print('adagrad He')
    # nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=epochs, learning_rate=0.01, mini_batch_size=50, weights_scale = 0.1, weights_loc = 0, momentum=1, weight_scale_type='he')
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='adagrad')


    # print()
    # print()
    # print()


    # print('momentum')
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum')
    # print('-------------')

    # print('momentum_nesterov')
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum_nesterov')
    # print('-------------')

    # print('adagrad')
    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='adagrad')
    # print('-------------')


    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum_nestrov')

    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig')

    # nn3.params = copy.deepcopy(params)
    # nn3.train_with_batch(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig', learning_rate_mode='momentum')


   
