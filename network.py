"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.acc_epoch = 0

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        """
        self.network = network

    def train(self, dataset, nepochs):
        """Train the network and record the accuracy.`

        """
        if self.accuracy == 0.:
            acc,a_e = train_and_score(self.network, dataset, nepochs)
            self.accuracy = acc
            self.acc_epoch = a_e

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%  acc_epoch=%d" % (self.accuracy * 100, self.acc_epoch))
