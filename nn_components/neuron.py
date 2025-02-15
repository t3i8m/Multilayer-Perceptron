import numpy as np

class Neuron(object):

    def __init__(self, prev_layer_neurons_size:int):
        """Constructor for the Neuron objects"""
        self.weights = np.random.randn(prev_layer_neurons_size, 1)

    def get_weights(self)->list:
        """Getter for the neuron`s weights [1 x prev_layer_neurons_size]"""
        return self.weights
    