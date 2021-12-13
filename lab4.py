import os

from tensorflow.python.keras.backend import dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # aby uniknąć osstrzeżeń związanym z GPU

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import numpy as np
import time

from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tqdm.keras import TqdmCallback



# train_size = 20000
# test_size = 10000
learning_rate = 0.001
verbose = 0
dropout_rate = 0.5

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

y_train = y_train.reshape(y_train.shape[0], 1)
y_train = keras.utils.to_categorical(y_train)

y_test = y_test.reshape(y_test.shape[0], 1)
y_test = keras.utils.to_categorical(y_test)

x_train = (x_train/255).astype('float64')
x_test = (x_test/255).astype('float64')
y_train = (y_train).astype('float64')
y_test = (y_test).astype('float64')




def mlp(epochs = 10, batch_size = 50, first_layer_size= 64):
    global x_train
    global x_test
    global y_train
    global y_test
    global verbose
    global learning_rate

    model = models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(first_layer_size, activation=activations.relu,
                                 kernel_initializer=initializers.GlorotNormal, # Xavier
                                 ))

    model.add(keras.layers.Dense(first_layer_size/2, activation=activations.relu,
                                 kernel_initializer=initializers.GlorotNormal))

    model.add(keras.layers.Dense(10, activation="softmax",
                                 kernel_initializer=initializers.GlorotNormal))


    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    start = time.time_ns()
    history = model.fit( # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=verbose,
        callbacks=[TqdmCallback(verbose=0)]
    )

    ti = (time.time_ns() - start) / 1000000000
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print("Final accuracy:", round(test_acc,6), "Total time:", round(ti,4), "s")

    return history

def mlp_dropout(epochs = 10, batch_size = 50, first_layer_size= 64):
    global x_train
    global x_test
    global y_train
    global y_test
    global verbose
    global learning_rate
    global dropout_rate

    model = models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate = dropout_rate))
    model.add(keras.layers.Dense(first_layer_size, activation=activations.relu,
                                 kernel_initializer=initializers.GlorotNormal, # Xavier
                                 ))

    model.add(keras.layers.Dense(first_layer_size/2, activation=activations.relu,
                                 kernel_initializer=initializers.GlorotNormal))

    model.add(keras.layers.Dense(10, activation="softmax",
                                 kernel_initializer=initializers.GlorotNormal))


    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    start = time.time_ns()
    history = model.fit( # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=verbose,
        callbacks=[TqdmCallback(verbose=0)]
    )

    ti = (time.time_ns() - start) / 1000000000
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print("Final accuracy:", round(test_acc,6), "Total time:", round(ti,4), "s")

    return history

def conv(conv_size=64, conv_layers_count = 2, kernel_size=(3,3), epochs = 10, batch_size = 50):
    global x_train
    global x_test
    global y_train
    global y_test
    global verbose
    global learning_rate

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28, 1)))

    current_conv_size = conv_size
    for i in range(conv_layers_count):
        model.add(keras.layers.Conv2D(current_conv_size, kernel_size=kernel_size,
                                    activation=activations.relu,
                                    kernel_initializer=initializers.GlorotNormal))
        current_conv_size = current_conv_size/2

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(10, activation="softmax",
                                 kernel_initializer=initializers.GlorotNormal))


    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    start = time.time_ns()
    history = model.fit(  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=verbose,
        callbacks=[TqdmCallback(verbose=0)]
    )

    ti = (time.time_ns() - start) / 1000000000
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("Final accuracy:", round(test_acc,6), "Total time:", round(ti,4), "s")

    return history


def conv_pooling(conv_size=64, conv_layers_count = 2, kernel_size=(3,3), pool_size=(2, 2), pool_type="max", epochs = 10, batch_size = 50):
    global x_train
    global x_test
    global y_train
    global y_test
    global verbose
    global learning_rate

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28, 1)))

    current_conv_size = conv_size
    for i in range(conv_layers_count):
        model.add(keras.layers.Conv2D(current_conv_size, kernel_size=kernel_size,
                                    activation=activations.relu,
                                    kernel_initializer=initializers.GlorotNormal))
        current_conv_size = current_conv_size - 20
        if pool_type == "max":
            model.add(layers.MaxPooling2D(pool_size=pool_size))
        elif pool_type == "avg":
            model.add(layers.AveragePooling2D(pool_size=pool_size))

    model.add(keras.layers.Flatten())


    model.add(keras.layers.Dense(10, activation="softmax",
                                 kernel_initializer=initializers.GlorotNormal))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    start = time.time_ns()
    history = model.fit( # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=verbose,
        callbacks=[TqdmCallback(verbose=0)]
    )

    ti = (time.time_ns() - start) / 1000000000
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("Final accuracy:", round(test_acc,6), "Total time:", round(ti,4), "s")

    return history

def main1():
    history = mlp()
    print('mlp')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))


    history = mlp_dropout()
    print('mlp_dropout')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))


    history = conv()
    print('conv')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))


    history = conv_pooling()
    print('conv_pooling')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

def main2():

    history = conv(80,1)
    print('conv')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

    history = conv(80,2)
    print('conv')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

    history = conv(80,3)
    print('conv')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

    history = conv(80,4)
    print('conv')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

def main3():
    print()
    print('2x2 pooling')
    history = conv_pooling(80, 2, (3,3), (2,2), 'max')
    print('conv_pooling max')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

    history = conv_pooling(80, 2, (3,3), (2,2), 'avg')
    print('conv_pooling avg')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

def main4():
    print()
    print('3x3 pooling')
    history = conv_pooling(80,2, (3,3), (3,3), 'max')
    print('conv_pooling max')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

    history = conv_pooling(80,2, (3,3), (3,3), 'avg')
    print('conv_pooling avg')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

def main5():
    print()
    print('4x4 pooling')
    history = conv_pooling(80,2, (3,3), (4,4), 'max')
    print('conv_pooling max')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

    history = conv_pooling(80,2, (3,3), (4,4), 'avg')
    print('conv_pooling avg')
    for accuracy in history.history['val_accuracy']:
        print((str(round(accuracy,4)).replace('.',',')))

if __name__ == "__main__":
    # main1()
    # main1()
    # main1()
    # main1()
    # main1()

    
    # main2()
    # main2()
    # main2()
    # main2()
    # main2()


    # main3()
    # main3()
    # main3()
    # main3()
    # main3()

    # print()
    # print()

    # main4()
    # main4()
    # main4()
    # main4()
    # main4()

    print()
    print()

    # main5()
    # main5()
    # main5()
    # main5()
    # main5()



### Plan badań ###

# 1. Porównianie sieci mlp z konwolucyjną (czas i accuracy na 10 epokach)
# MLP: Wejściowa -> ukryta(64) -> ukryta(32) -> wyjściowa
# MLP_dropout: Wejściowa -> ukryta(64) -> ukryta(32) -> wyjściowa
# Konwolucyjna: Wejściowa -> Konwolucyjna(64) -> Konwolucyjna(32) -> wyjściowa
# Konwolucyjna z pooling: Wejściowa -> Konwolucyjna(64) -> pooling(max) -> Konwolucyjna(32) -> wyjściowa 

# 2. Porównianie wpływu ilości warstw konwolucyjnych na wynik (4/3/2) (czas i accuracy na 10 epokach)
# Konwolucyjna: Wejściowa -> Konwolucyjna(64) -> wyjściowa
# Konwolucyjna: Wejściowa -> Konwolucyjna(64) -> Konwolucyjna(32) -> wyjściowa
# Konwolucyjna: Wejściowa -> Konwolucyjna(64) -> Konwolucyjna(32) -> Konwolucyjna(16) -> wyjściowa

# 3.Badania Pooling
# 3.1 Porównanie pooling'u (max i avg) (czas i accuracy na 10 epokach) wymiary(2,2)
# Konwolucyjna z pooling: Wejściowa -> Konwolucyjna(64) -> pooling -> Konwolucyjna(32) -> pooling -> wyjściowa

# 3.2 Porównanie pooling'u (max i avg) (czas i accuracy na 10 epokach) wymiary(3,3)
# Konwolucyjna z pooling: Wejściowa -> Konwolucyjna(64) -> pooling -> Konwolucyjna(32) -> pooling -> wyjściowa