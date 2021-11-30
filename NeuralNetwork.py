import numpy as np
import time
import copy
import random
import math 
  

from numpy.core.fromnumeric import size

class NeuralNetwork:
    def __init__(self, sizes, epochs=100, learning_rate=0.1, mini_batch_size = 4, weights_scale = 0.1, weights_loc = 0, early_stop_value = 0.05, 
                momentum = 0.7, weight_scale_type ='normal', ad_gamma= 0.9, beta_1 =0.9, beta_2 = 0.999):
        # Tablica z ilością neuronów na poszczególnych wartstwach
        # Tutaj zastosowałem 4 warstwy, wejściową, 2 ukryte i wyjściową
        self.sizes = sizes
        self.layer_count = len(sizes)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        
        # Parametry - przechowuje wagi oraz bias'y, oraz później poszczególne

        self.params = self.initialize(weights_scale, weights_loc, weight_scale_type)

        self.early_stop_value = early_stop_value
        self.best_error = 1
        self.best_params = copy.deepcopy(self.params)


        self.momentum = momentum

        self.ad_gamma = ad_gamma

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 1



    def initialize(self, weights_scale = 0.1, weights_loc =0, weight_scale_type = 'normal'):
        # Przypisanie wielkości poszególnych warstw
        input_layer_size = self.sizes[0]
        hidden_layers_sizes = []

        for i in range(1, len(self.sizes)-1):
            hidden_layers_sizes.append(self.sizes[i])

        output_layer_size = self.sizes[len(self.sizes)-1]

        params = {}

        if(hidden_layers_sizes != []):
            # Xavier
            if weight_scale_type == 'xavier':
                weights_scale = np.sqrt(2 / (hidden_layers_sizes[0] + input_layer_size))
            # HE
            elif weight_scale_type == 'he':
                weights_scale = np.sqrt(2 / (input_layer_size))
            
            params['W1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[0], input_layer_size)) 
            params['PW1'] = np.zeros( shape=(hidden_layers_sizes[0], input_layer_size)) 
            params['GSW1'] = np.zeros( shape=(hidden_layers_sizes[0], input_layer_size)) 
            params['MNW1'] = np.zeros( shape=(hidden_layers_sizes[0], input_layer_size)) 
            params['AGAW1'] = np.zeros( shape=(hidden_layers_sizes[0], input_layer_size)) 
            params['ADAW1'] = np.zeros( shape=(hidden_layers_sizes[0], input_layer_size))
            params['MPW1'] = np.zeros( shape=(hidden_layers_sizes[0], input_layer_size)) 
            params['VPW1'] = np.zeros( shape=(hidden_layers_sizes[0], input_layer_size)) 

            params['B1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[0]))
            params['PB1'] = np.zeros( shape=(hidden_layers_sizes[0])) 
            params['GSB1'] = np.zeros( shape=(hidden_layers_sizes[0])) 
            params['MNB1'] = np.zeros( shape=(hidden_layers_sizes[0])) 
            params['AGAB1'] = np.zeros( shape=(hidden_layers_sizes[0])) 
            params['ADAB1'] = np.zeros( shape=(hidden_layers_sizes[0]))
            params['MPB1'] = np.zeros( shape=(hidden_layers_sizes[0])) 
            params['VPB1'] = np.zeros( shape=(hidden_layers_sizes[0]))  
        else:
            # Xavier
            if weight_scale_type == 'xavier':
                weights_scale = np.sqrt(2 / (output_layer_size + input_layer_size))
            # He
            elif weight_scale_type == 'he':
                weights_scale = np.sqrt(2 / (input_layer_size))

            params['W1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size, input_layer_size))
            params['PW1'] = np.zeros( shape=(output_layer_size, input_layer_size)) 
            params['GSW1'] = np.zeros( shape=(output_layer_size, input_layer_size)) 
            params['MNW1'] = np.zeros( shape=(output_layer_size, input_layer_size)) 
            params['AGAW1'] = np.zeros( shape=(output_layer_size, input_layer_size)) 
            params['ADAW1'] = np.zeros( shape=(output_layer_size, input_layer_size)) 
            params['MPW1'] = np.zeros( shape=(output_layer_size, input_layer_size)) 
            params['VPW1'] = np.zeros( shape=(output_layer_size, input_layer_size)) 

            params['B1'] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size))
            params['PB1'] = np.zeros( shape=(output_layer_size)) 
            params['GSB1'] = np.zeros( shape=(output_layer_size)) 
            params['MNB1'] = np.zeros( shape=(output_layer_size)) 
            params['AGAB1'] = np.zeros( shape=(output_layer_size)) 
            params['ADAB1'] = np.zeros( shape=(output_layer_size))
            params['MPB1'] = np.zeros( shape=(output_layer_size)) 
            params['VPB1'] = np.zeros( shape=(output_layer_size))

        for i in range(len(hidden_layers_sizes)):
            if(i < len(hidden_layers_sizes)-1):
                # Xavier
                if weight_scale_type == 'xavier':
                    weights_scale = np.sqrt(2 / (hidden_layers_sizes[i+1] + hidden_layers_sizes[i]))
                # He
                elif weight_scale_type == 'he':
                    weights_scale = np.sqrt(2 / (hidden_layers_sizes[i]))

                params['W'+str(i+2)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[i+1], hidden_layers_sizes[i]))
                params['PW'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1], hidden_layers_sizes[i])) 
                params['GSW'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1], hidden_layers_sizes[i])) 
                params['MNW'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1], hidden_layers_sizes[i]))
                params['AGAW'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1], hidden_layers_sizes[i])) 
                params['ADAW'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1], hidden_layers_sizes[i]))
                params['MPW'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1], hidden_layers_sizes[i])) 
                params['VPW'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1], hidden_layers_sizes[i]))

                params['B'+str(i+2)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(hidden_layers_sizes[i+1]))
                params['PB'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1])) 
                params['GSB'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1])) 
                params['MNB'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1])) 
                params['AGAB'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1])) 
                params['ADAB'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1])) 
                params['MPB'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1])) 
                params['VPB'+str(i+2)] = np.zeros( shape=(hidden_layers_sizes[i+1])) 
            else:
                # Xavier
                if weight_scale_type == 'xavier':
                    weights_scale = np.sqrt(2 / (output_layer_size + hidden_layers_sizes[i]))
                # He
                elif weight_scale_type == 'he':
                    weights_scale = np.sqrt(2 / (hidden_layers_sizes[i]))

                params['W'+str(i+2)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size, hidden_layers_sizes[i]))
                params['PW'+str(i+2)] = np.zeros( shape=(output_layer_size, hidden_layers_sizes[i])) 
                params['GSW'+str(i+2)] = np.zeros( shape=(output_layer_size, hidden_layers_sizes[i])) 
                params['MNW'+str(i+2)] = np.zeros( shape=(output_layer_size, hidden_layers_sizes[i]))
                params['AGAW'+str(i+2)] = np.zeros( shape=(output_layer_size, hidden_layers_sizes[i])) 
                params['ADAW'+str(i+2)] = np.zeros( shape=(output_layer_size, hidden_layers_sizes[i])) 
                params['MPW'+str(i+2)] = np.zeros( shape=(output_layer_size, hidden_layers_sizes[i])) 
                params['VPW'+str(i+2)] = np.zeros( shape=(output_layer_size, hidden_layers_sizes[i])) 

        # Xavier
        if weight_scale_type == 'xavier':
            weights_scale = np.sqrt(2 / (output_layer_size))
        # He
        elif weight_scale_type == 'he':
            weights_scale = np.sqrt(2 / (output_layer_size))

        params['B'+str(len(hidden_layers_sizes)+1)] = np.random.normal(loc = weights_loc, scale=weights_scale, size=(output_layer_size))
        params['PB'+str(len(hidden_layers_sizes)+1)] = np.zeros( shape=(output_layer_size)) 
        params['GSB'+str(len(hidden_layers_sizes)+1)] = np.zeros( shape=(output_layer_size))
        params['MNB'+str(len(hidden_layers_sizes)+1)] = np.zeros( shape=(output_layer_size))
        params['AGAB'+str(len(hidden_layers_sizes)+1)] = np.zeros( shape=(output_layer_size))
        params['ADAB'+str(len(hidden_layers_sizes)+1)] = np.zeros( shape=(output_layer_size))
        params['MPB'+str(len(hidden_layers_sizes)+1)] = np.zeros( shape=(output_layer_size))
        params['VPB'+str(len(hidden_layers_sizes)+1)] = np.zeros( shape=(output_layer_size))

        return params

    def forward(self, x_train, bias_active=True, sigmoid_active=True, tanhip_active=True, momentum_nesterov = False):
        params = self.params

        cl = 0
        # Wejściowa warstwa
        params['A0'] = x_train

        # Wejściowa warstwa -> Pierwsza warstwa ukryta
        for i in range(self.layer_count-2):
            cl += 1
            cls = str(cl)
            pls = str(cl-1)

            if not momentum_nesterov:
                weight = params['W' + cls]
                weight_b = params['B' + cls]
            else:
                weight = params['MNW' + cls]
                weight_b = params['MNB' + cls]

            if(bias_active):
                params['Z'+cls] = np.add(np.dot(weight, params['A'+pls]), weight_b)
            else:
                params['Z'+cls] = np.dot(weight, params['A'+pls])

            if(sigmoid_active):
                params['A'+cls] = self.sigmoid(params['Z'+cls])
            elif(tanhip_active):
                params['A'+cls] = self.tanhip(params['Z'+cls])
            else:
                params['A'+cls] = self.ReLU(params['Z'+cls])

        cl += 1
        cls = str(cl)
        pls = str(cl-1)

        if not momentum_nesterov:
            weight = params['W' + cls]
            weight_b = params['B' + cls]
        else:
            weight = params['MNW' + cls]
            weight_b = params['MNB' + cls]

        if(bias_active):
            params['Z'+cls] = np.add(np.dot(weight, params['A'+pls]), weight_b)
        else:
            params['Z'+cls] = np.dot(weight, params['A'+pls]) 

        params['A'+cls] = self.softmax(params['Z'+cls])

        return params['A'+cls]

    def backward(self, y_train, output, sigmoid_active=True, tanhip_active=True, momentum_nesterov = False):
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

            if not momentum_nesterov:
                weight_t = params['W' + nls].T
            else:
                weight_t = params['MNW' + nls].T
            

            if(sigmoid_active):
                error = np.dot(weight_t, error) * self.sigmoid(params['Z' + cls], derivative=True)
            elif(tanhip_active):
                error = np.dot(weight_t, error) * self.tanhip(params['Z' + cls], derivative=True)
            else:
                error = np.dot(weight_t, error) * self.ReLU(params['Z' + cls], derivative=True)

            change_weights['W' + cls] = np.outer(error, params['A' + pls])
            change_weights['B'+ cls] = error

        return change_weights

    def update_network_parameters(self, changes_to_w, learning_rate_mode = 'normal', epsilon=1e-8):
        for key, value in changes_to_w.items():
            if learning_rate_mode == 'momentum':
                self.params['P' + key] = self.learning_rate * value + self.momentum * self.params['P' + key]
                self.params[key] -= self.params['P' + key]

            elif learning_rate_mode == 'momentum_nesterov':
                self.params['P' + key] = self.learning_rate * value + self.momentum * self.params['P' + key]
                self.params[key] -= self.params['P' + key]
                self.params['MN' + key] = self.params[key] - self.momentum * self.params['P' + key]

            elif learning_rate_mode == 'adagrad':
                self.params['GS' + key] += (value) ** 2
                self.params[key] -= (self.learning_rate * value) / (np.sqrt(self.params['GS' + key] + epsilon))

            elif learning_rate_mode == 'adelta':
                self.params['AGA' + key] = (self.ad_gamma * self.params['AGA' + key]) + (1 - self.ad_gamma) * np.square(value)
                new_value = (np.sqrt(self.params['ADA' + key] + epsilon)) / (np.sqrt(self.params['AGA' + key] + epsilon)) * value
                self.params['ADA' + key] = (self.ad_gamma * self.params['ADA' + key]) + (1 - self.ad_gamma) * np.square(new_value)
                self.params[key] -= new_value

            elif learning_rate_mode == 'adam':
                self.params['MP' + key] = self.beta_1 * self.params['MP' + key] + (1-self.beta_1) * value 
                self.params['VP' + key] = self.beta_2 * self.params['VP' + key] + (1-self.beta_2) * np.square(value)

                m_corrected = self.params['MP' + key] / (1-self.beta_1**self.t)
                v_corrected = self.params['VP' + key] / (1-self.beta_2**self.t)

                self.params[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8) 
            else:
                # default: normal
                self.params[key] -= self.learning_rate * value

        self.t+=1



    def compute_accuracy(self, x_val, y_val):
        predictions_sum = 0
        for x, y in zip(x_val, y_val):
            output = self.forward(x)
            pred = np.argmax(output)
            if(pred == y):
                predictions_sum += 1

        return predictions_sum/len(y_val)

    def train_with_batch(self, x_train, y_train, x_val, y_val, bias_active = True, activation_function = 'sig', learning_rate_mode = 'normal', momentum_nesterov = False):
        if(activation_function == 'sig'):
            sigmoid_active = True
            tanhip_active = False
        elif(activation_function == 'tan'):
            sigmoid_active = False
            tanhip_active = True
        else:
            sigmoid_active = False
            tanhip_active = False

        if learning_rate_mode == 'momentum_nesterov':
            momentum_nesterov = True
        else:
            momentum_nesterov = False

        training_size = x_train.shape[0]
        start_time = time.time() 
        for iteration in range(self.epochs):
            c = list(zip(x_train, y_train))
            random.shuffle(c)
            x_train, y_train = zip(*c)

            for i in range(0, training_size, self.mini_batch_size):
                output = self.forward(x_train[i], bias_active, sigmoid_active, tanhip_active, momentum_nesterov)
                change = self.backward(y_train[i], output, sigmoid_active, tanhip_active, momentum_nesterov)
   
                for x, y in zip(x_train[i+1 : min(i+self.mini_batch_size, training_size -1)], y_train[i+1 : min(i+self.mini_batch_size, training_size -1)]):
                    output = self.forward(x, bias_active, sigmoid_active, tanhip_active, momentum_nesterov)
                    next_change = self.backward(y, output, sigmoid_active, tanhip_active, momentum_nesterov)
                    change = {key: np.add(change[key], next_change[key])  for key in change.keys()}

                change = {key: change[key]/self.mini_batch_size for key in change.keys()}

                self.update_network_parameters(change, learning_rate_mode)
            
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
