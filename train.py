"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)   # 2 or 5

def compile_model(network, nb_classes, nb_features):
    """
    Compile a sequential model.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers+1):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation='sigmoid', kernel_initializer='glorot_normal', input_dim=nb_features))
        else:
            model.add(Dense(nb_neurons, activation=activation, kernel_initializer='glorot_normal'))

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def read_samples(file_name):
#    print('\nReading samples ...', file_name)
    df = pd.read_csv(file_name)
    X = df.iloc[:, 1:].values
    X = X.astype(np.float32)

    kinx = df.columns.get_loc('kernel')
    if kinx != 0:
        print('Warning: The logic assumes the kernel attribute is index 0')
    dfy = df.loc[:,'kernel'].values

    encoder = preprocessing.LabelEncoder()
    encoder.fit(dfy)
#    print('Encoder classes = \n')
#    for c in encoder.classes_:
#        print('\t', c)

    encoded_Y = encoder.transform(dfy)
    encoded_Y = encoded_Y.astype(np.float32)
    onehot_Y = np_utils.to_categorical(encoded_Y)
    onehot_Y = onehot_Y.astype(np.float32)
    return X,onehot_Y,encoded_Y

def train_and_score(network, dataset, nepochs):
    """
    Train the model and return test loss.

    """

    x_train, y_train, none  = read_samples(dataset['train'])
    x_test, y_test, none = read_samples(dataset['test'])
    nb_classes = 8
    nb_features = 90 - 1

    model = compile_model(network, nb_classes, nb_features)

    batchsize = network['batchsize']

    history = model.fit(x_train, y_train,
              batch_size=batchsize,
              epochs=nepochs,
              verbose=0,
              validation_data=(x_test, y_test))
#              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1],len(history.history['acc'])  # 1 is accuracy. 0 is loss.
