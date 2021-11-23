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
    train_size = 20000
    test_size = 2000
    nn3 = NeuralNetwork(sizes=[784, 128, 10], epochs=100, learning_rate=0.01, mini_batch_size=30, weights_scale = 0.1, weights_loc = 0)
    nn3.train(x_train[0:train_size], y_train[0:train_size], x_text[0:test_size], y_test[0:test_size],activation_function='sig')