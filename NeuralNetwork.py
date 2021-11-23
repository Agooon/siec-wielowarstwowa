import numpy as np
import time
import copy
import random

from numpy.core.fromnumeric import size

class NeuralNetwork:
    def __init__(self, sizes, epochs=100, learning_rate=0.1, mini_batch_size = 4, weights_scale = 0.1, weights_loc = 0, early_stop_value = 0.05):
        # Tablica z ilością neuronów na poszczególnych wartstwach
        # Tutaj zastosowałem 4 warstwy, wejściową, 2 ukryte i wyjściową
        self.sizes = sizes
        self.layer_count = len(sizes)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        
        # Parametry - przechowuje wagi oraz bias'y, oraz później poszczególne
        self.params = self.initialize(weights_scale, weights_loc)

        self.early_stop_value = early_stop_value
        self.best_error = 1
        self.best_params = copy.deepcopy(self.params)

    def initialize(self, weights_scale = 0.1, weights_loc =0):
        # Przypisanie wielkości poszególnych warstw
        input_layer_size = self.sizes[0]
        hidden_layers_sizes = []

        for i in range(1, len(self.sizes)-1):
            hidden_layers_sizes.append(self.sizes[i])

        output_layer_size = self.sizes[len(self.sizes)-1]

        params = {}

        if(hidden_layers_sizes != []):
            params['W1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[0], input_layer_size)) 
            params['B1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[0])) 
        else:
            params['W1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size, input_layer_size)) 
            params['B1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size)) 

        for i in range(len(hidden_layers_sizes)):
            if(i < len(hidden_layers_sizes)-1):
                params['W'+str(i+2)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[i+1], hidden_layers_sizes[i])) 
                params['B'+str(i+2)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[i+1])) 
            else:
                params['W'+str(i+2)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size, hidden_layers_sizes[i])) 

        params['B'+str(len(hidden_layers_sizes)+1)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size)) 

        return params

    def forward(self, x_train, bias_active=True, sigmoid_active=True, tanhip_active=True):
        params = self.params

        cl = 0
        # Wejściowa warstwa
        params['A0'] = x_train

        # Wejściowa warstwa -> Pierwsza warstwa ukryta
        for i in range(self.layer_count-2):
            cl += 1
            cls = str(cl)
            pls = str(cl-1)
            if(bias_active):
                params['Z'+cls] = np.add(np.dot(params["W" + cls], params['A'+pls]), params['B'+cls])
            else:
                params['Z'+cls] = np.dot(params["W"+cls], params['A'+pls])

            if(sigmoid_active):
                params['A'+cls] = self.sigmoid(params['Z'+cls])
            elif(tanhip_active):
                params['A'+cls] = self.tanhip(params['Z'+cls])
            else:
                params['A'+cls] = self.ReLU(params['Z'+cls])

        cl += 1
        cls = str(cl)
        pls = str(cl-1)
        if(bias_active):
            params['Z'+cls] = np.add(np.dot(params["W"+cls], params['A'+pls]), params['B'+cls])
        else:
            params['Z'+cls] = np.dot(params["W"+cls], params['A'+pls])

        params['A'+cls] = self.softmax(params['Z'+cls])

        return params['A'+cls]

    def backward(self, y_train, output, sigmoid_active=True, tanhip_active=True):
        params = self.params
        change_weights = {}
        cl = self.layer_count - 1
        cls = str(cl)
        pls = str(cl-1)
        nls = str(cl+1)

        index = y_train
        y_train = np.zeros(10)
        y_train[index] = 1
        # Obliczamy ostatnią 
        error = (output - y_train) * self.softmax(params['Z'+cls], derivative=True)
        change_weights['W'+cls] = np.outer(error, params['A'+pls])
        change_weights['B'+cls] = error

        for i in range(self.layer_count - 1,1,-1):
            cl -= 1
            cls = str(cl)
            pls = str(cl-1)
            nls = str(cl+1)

            if(sigmoid_active):
                error = np.dot(params['W' + nls].T, error) * self.sigmoid(params['Z' + cls], derivative=True)
            elif(tanhip_active):
                error = np.dot(params['W' + nls].T, error) * self.tanhip(params['Z' + cls], derivative=True)
            else:
                error = np.dot(params['W' + nls].T, error) * self.ReLU(params['Z' + cls], derivative=True)

            change_weights['W' + cls] = np.outer(error, params['A' + pls])
            change_weights['B'+ cls] = error

        return change_weights

    def update_network_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.params[key] -= self.learning_rate * value


    def compute_accuracy(self, x_val, y_val):
        predictions_sum = 0
        for x, y in zip(x_val, y_val):
            output = self.forward(x)
            pred = np.argmax(output)
            if(pred == y):
                predictions_sum += 1

        return predictions_sum/len(y_val)

    def train_with_batch(self, x_train, y_train, x_val, y_val, bias_active = True, activation_function = 'sig'):
        if(activation_function == 'sig'):
            sigmoid_active = True
            tanhip_active = False
        elif(activation_function == 'tan'):
            sigmoid_active = False
            tanhip_active = True
        else:
            sigmoid_active = False
            tanhip_active = False

        training_size = x_train.shape[0]
        start_time = time.time() 
        for iteration in range(self.epochs):
            c = list(zip(x_train, y_train))
            random.shuffle(c)
            x_train, y_train = zip(*c)

            for i in range(0, training_size, self.mini_batch_size):
                output = self.forward(x_train[i], bias_active, sigmoid_active, tanhip_active)
                change = self.backward(y_train[i], output, sigmoid_active, tanhip_active)
   
                for x, y in zip(x_train[i+1 : min(i+self.mini_batch_size, training_size -1)], y_train[i+1 : min(i+self.mini_batch_size, training_size -1)]):
                    output = self.forward(x, bias_active, sigmoid_active, tanhip_active)
                    next_change = self.backward(y, output, sigmoid_active, tanhip_active)
                    change = {key: np.add(change[key], next_change[key])  for key in change.keys()}

                change = {key: change[key]/self.mini_batch_size for key in change.keys()}

                self.update_network_parameters(change)
            
            accuracy = self.compute_accuracy(x_val, y_val)
            # print('[Training_size: {5}][Batch: {4}][{3}] Epoka: {0}, Czas: {1:.2f}s, Procent dokłądności: {2:.2f}%'.format(
            #     iteration+1, time.time() - start_time, accuracy * 100, activation_function, self.mini_batch_size, training_size
            # ))
            print('{0};{1}'.format(
                iteration+1, str(accuracy*100).replace('.', ',')
            ))
            # self.checkEarlyStop(1- accuracy)

    def train(self, x_train, y_train, x_val, y_val, bias_active = True, activation_function = 'sig'):
        if(activation_function == 'sig'):
            sigmoid_active = True
            tanhip_active = False
        elif(activation_function == 'tan'):
            sigmoid_active = False
            tanhip_active = True
        else:
            sigmoid_active = False
            tanhip_active = False

        training_size = x_train.shape[0]
        start_time = time.time() 
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward(x, bias_active, sigmoid_active, tanhip_active)
                change = self.backward(y, output, sigmoid_active, tanhip_active)
                self.update_network_parameters(change)
            
            accuracy = self.compute_accuracy(x_val, y_val)
            # print('[Training_size: {4}][{3}] Epoka: {0}, Czas: {1:.2f}s, Procent dokłądności: {2:.2f}%'.format(
            #     iteration+1, time.time() - start_time, accuracy * 100, activation_function, training_size
            # ))
            print('{0};{1}'.format(
                iteration+1, str(accuracy*100).replace('.', ',')
            ))
            # self.checkEarlyStop(1- accuracy)

    #########################
    ### Funkcje aktywacji ###
    #########################
    def sigmoid(self, x, derivative = False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        else:
            return 1/(1 + np.exp(-x))

    def tanhip(self, x, derivative = False):
        if(derivative):
            return 1.0 - np.tanh(x)**2
        else:
            return np.tanh(x)

    def ReLU(self, x, derivative = False):
        if (derivative):
            return np.maximum(np.sign(x) == 1, 0)
        else:
            return np.maximum(0, x)

    def softmax(self, x, derivative = False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def checkEarlyStop(self, error):
        if(self.best_error - error < self.early_stop_value):
            print('EARLY STOP')
            self.params = copy.deepcopy(self.best_params)
        elif(self.best_error > error):
            self.best_error = error
            self.best_params = copy.deepcopy(self.params)
