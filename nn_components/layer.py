import numpy as np
from nn_components.neuron import Neuron

class Layer(object):

    def __init__(self, num_neurons:int, prev_layer_size:int):
        """Constructor for the layer`s objects"""
        self.size = num_neurons
        self.prev_layer_size = prev_layer_size
        self.initialize_weights_biases()

    def get_size(self)->int:
        """Getter for the number of neurons in a particular layer"""
        return self.size

    def get_weights(self)->np.array:
        """Getter for the weights, return a matrix [num_neurons x prev_layer_size]"""
        return [n.get_weights() for n in self.neurons]
    
    def get_biases(self)->np.array:
        """Getter for the biases, return a matrix [num_neurons x 1]"""
        return [n.get_bias() for n in self.neurons]

    def initialize_weights_biases(self)->None:
        """Randomly initializes neurons and corresponding parameters """
        if(self.prev_layer_size==0):
            return

        self.neurons = [Neuron(self.prev_layer_size, bias) for _, bias in zip(range(self.size), np.random.randn(self.size, 1))]
        return
    
    def set_activations(self, activations:np.array)->None:
        """Setter for the activations"""
        self.activations = activations
        return 
    
    def get_activations(self)->np.array:
        """Getter for the activations"""
        return self.activations
    
    def set_gradients_weights(self, gradients:np.array, new=False)->None:
        """Setter for the gradients"""
        for neuron, gradients in zip(self.neurons, gradients):
            neuron.add_gradients_weights(gradients)

        return 
    
    
    def set_gradients_bias(self, gradients:np.array)->None:
        """Setter for the gradients"""
        for neuron, gradients in zip(self.neurons, gradients):
            neuron.add_gradients_biases(gradients)

        return 
    
    def update_weights(self, learning_rate:float)->None:
        """Updating weights after one minibatch"""
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)
        return
    
    def update_biases(self, learning_rate:float)->None:
        """Updating biases after one minibatch"""
        for neuron in self.neurons:
            neuron.update_biase(learning_rate)
        return

    
