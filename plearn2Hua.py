#
# plearn2Hua.py
#
# This python file will either train a model or create a new model
# using a genetic algorithm.
#
# Bob Zigon, rzigon@iupui.edu
#
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from keras.utils import np_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import logging
from tqdm import tqdm
from optimizer import Optimizer


def create_base_network(NumberOfFeatures, NumberOfClasses,init_mode='glorot_normal'):
    """
    This function will create the base network and return the model.

    This is a helper function that the other network creation functions will call.
    """
    network = Sequential()
    network.add(Dense(NumberOfFeatures, activation='sigmoid', kernel_initializer=init_mode,input_dim=NumberOfFeatures))
    network.add(Dense(NumberOfFeatures, activation='relu',kernel_initializer=init_mode))
    network.add(Dense(NumberOfClasses, activation='softmax',kernel_initializer=init_mode))
    return network


def create_network(NumberOfFeatures, NumberOfClasses, optimizer_type, lr, moment, lr_decay):
    """
    This function will create the network and return the model.

    """
    model = create_base_network(NumberOfFeatures, NumberOfClasses)
    if optimizer_type == 'sgd':
        opt = optimizers.SGD(lr=lr, momentum=moment, decay=lr_decay)
    else:
        opt = optimizer_type

    model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    return model


def read_samples(file_name):
    print('\nReading samples ...', file_name)
    df = pd.read_csv(file_name)
    X = df.iloc[:, 1:].values
    X = X.astype(np.float32)

    kinx = df.columns.get_loc('kernel')
    if kinx != 0:
        print('Warning: The logic assumes the kernel attribute is index 0')
    dfy = df.loc[:,'kernel'].values

    encoder = preprocessing.LabelEncoder()
    encoder.fit(dfy)
    print('Encoder classes = \n')
    for c in encoder.classes_:
        print('\t', c)

    encoded_Y = encoder.transform(dfy)
    encoded_Y = encoded_Y.astype(np.float32)
    onehot_Y = np_utils.to_categorical(encoded_Y)
    onehot_Y = onehot_Y.astype(np.float32)
    return X,onehot_Y,encoded_Y


def do_train(NumberOfFeatures, NumberOfClasses, optimizer_type, lr, momentum, lr_decay, cb_list, train_input_file, test_input_file, batch_size, max_epochs):

    model = create_network(NumberOfFeatures, NumberOfClasses, optimizer_type, lr, momentum, lr_decay)

    train_x, train_y, none  = read_samples(train_input_file)
    test_x, test_y, none = read_samples(test_input_file)

    history = model.fit(train_x, train_y,
                        epochs=max_epochs,
                        verbose=2,
                        batch_size=batch_size,
                        callbacks=cb_list,
                        validation_data=(test_x, test_y))  # Data to use for evaluation
    return history, model


def graph_model(history, model, file_name, title):
    training_accuracy = history.history['acc']
    test_accuracy = history.history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_accuracy) + 1)

    # Visualize accuracy history
    plt.plot(epoch_count, training_accuracy, 'r--')
    plt.plot(epoch_count, test_accuracy, 'b-')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.savefig(file_name)
    plt.ylim([0.4, 1.0])
    plt.show();
    for i in range(len(training_accuracy)):
        print('%3d train %8.6f  test %8.6f' % (i,training_accuracy[i],test_accuracy[i]))


def perform_training(mdlname, batch_size, nepochs, otype, lr, momentum, lr_decay):
    input_train_file = 'train' + train_suffix + '.txt'
    full_input_train_file = os.path.join(data_dir, input_train_file)

    input_test_file = 'test' + test_suffix + '.txt'
    full_input_test_file = os.path.join(data_dir, input_test_file)

    callback_list = []
    history, predictive_model = do_train(nFeatures,
                                         nClasses,
                                         otype,
                                         lr,
                                         momentum,
                                         lr_decay,
                                         callback_list,
                                         full_input_train_file,
                                         full_input_test_file,
                                         batch_size,
                                         nepochs)

    graph_model(history, predictive_model, 'TrainTest-' + optimizer_type + '.pdf', 'Optimizer: ' + optimizer_type)
    predictive_model.save(mdlname)
    return


def perform_validation(mdlname):
    print('Loading model ... ' + mdlname)
    predictive_model = load_model(mdlname)

    valid_test_file = 'data' + valid_suffix + '.txt'
    full_valid_test_file = os.path.join(data_dir, valid_test_file)

    val_x, val_y, enc_y = read_samples(full_valid_test_file)
    classes = predictive_model.predict(val_x)

    for i in range(len(classes)):
        v = list('0' * nClasses)
        c = np.argmax(classes[i])
        v[c] = '1'
        print('%4d: ' % (i), '  '.join(v), ' pred class=%d' % (c), end="\n")
        #
        # The next print statement only makes sense if the validation file is the test file.
        # Otherwise, you dont know the true class of the rows in the data file.
        #
        #            print('  act class=%d' % (int(enc_y[i])))
    return

def train_networks(networks, datasets, nepochs):
    """
    Train each network.

    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(datasets, nepochs)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """
    Get the average accuracy for a group of networks.

    Returns the average accuracy of a population of networks.
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def print_networks(networks, msg):
    """
    Print a list of networks.

    """

    str = '-'*40
    str += msg
    str += '-'*40
    logging.info(str)
    for network in networks:
        network.print_network()

def generate(generations, population, nn_param_choices, datasets, nepochs):
    """
    Generate a network with the genetic algorithm.

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" % (i + 1, generations))
        print("***Doing generation %d of %d***" % (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, datasets, nepochs)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        print_networks(networks[:5], "")
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:25], "Final Results")
    print("Output is log.txt")


def perform_genetic_evolution(datasets):
    generations = 3     # Note: atleast 3 generations
    population = 8      # atleast a population of 8
    nepochs = 5

    nn_param_choices = {
        'nb_neurons': [10, 20,  40, 80],
        'nb_layers': [1, 2, 3],         # hidden layers
        'activation': ['relu',  'tanh', 'sigmoid'],
        'optimizer': ['adam', 'adagrad','adadelta', 'nadam'],
        'batchsize' : [16, 24, 32, 40, 48]
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, datasets, nepochs)

    return

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

######################################################################################################
##
######################################################################################################

if __name__ == '__main__':

#    optimizer_type = 'adadelta'
#    optimizer_type = 'sgd'
#    optimizer_type = 'adagrad'
#    optimizer_type = 'adam'
#    optimizer_type = 'rmsprop'
    optimizer_type = 'nadam'

    modelname = 'gpumodel.'+optimizer_type
    data_dir = "Huawei"
    NumberOfEpochs = 90
    batch_size = 32
    nFeatures = 90-1            # This is fixed for our input
    nClasses = 8
    learning_rate = 0.020       # learning rate for SGD
    learning_rate_decay = learning_rate / NumberOfEpochs


    train_suffix = '-42'        # Dont change this
    test_suffix = '-42'         # Dont change this
    valid_suffix = '-mmval'     # Dont change this

    if True:
        perform_training(modelname, batch_size, NumberOfEpochs, optimizer_type, learning_rate, 0.90, learning_rate_decay)
        perform_validation(modelname)


    # Note : LOG.TXT has the output in it when you run the genetic evolution.
    # This currently takes about 5 hours on a TitanV.
    if False:
        input_train_file = 'train' + train_suffix + '.txt'
        full_input_train_file = os.path.join(data_dir, input_train_file)

        input_test_file = 'test' + test_suffix + '.txt'
        full_input_test_file = os.path.join(data_dir, input_test_file)

        datasets = {
            'train': full_input_train_file,
            'test': full_input_test_file
        }

        # Set all the parameters in the body of the function perform_genetic_evolution.
        perform_genetic_evolution(datasets)




