import numpy as np
from nn_components.neuron import Neuron

class Layer(object):

    def __init__(self, num_neurons:int, prev_layer_size:int):
        """Constructor for the layer`s objects"""
        self.size = num_neurons
        self.prev_layer_size = prev_layer_size
        self.biases = np.random.randn(num_neurons, 1)
        self.initialize_weights_biases()

    def get_size(self)->int:
        """Getter for the number of neurons in a particular layer"""
        return self.size

    def get_weights(self)->np.array:
        """Getter for the weights, return a matrix [num_neurons x prev_layer_size]"""
        return [n.get_weights() for n in self.neurons]
    
    def get_biases(self)->np.array:
        """Getter for the biases, return a matrix [num_neurons x 1]"""
        return self.biases

    def initialize_weights_biases(self)->None:
        """Randomly initializes neurons and corresponding parameters """
        if(self.prev_layer_size==0):
            return

        self.neurons = [Neuron(self.prev_layer_size) for _ in range(self.size)]
        return