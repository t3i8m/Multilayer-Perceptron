import numpy as np

class Neuron(object):

    def __init__(self, prev_layer_neurons_size:int, bias:float):
        """Constructor for the Neuron objects"""
        self.weights = np.random.randn(prev_layer_neurons_size, 1)
        self.gradients_holder_weights = []
        self.gradients_holder_biases= []
        self.bias = bias

    def get_weights(self)->list:
        """Getter for the neuron`s weights [1 x prev_layer_neurons_size]"""
        return self.weights
    
    def add_gradients_weights(self, gradients:np.array)->None:
        """Add gradients for all of the weights in this neuron, input:[1 x prev_layer_neurons_size]"""
        self.gradients_holder_weights.append(gradients)
        return
    
    def add_gradients_biases(self, gradients:np.array)->None:
        """Add gradients for all of the biases in this neuron, input:[1 x prev_layer_neurons_size]"""
        self.gradients_holder_biases.append(gradients)
        return
    
    def update_weights(self, learning_rate:float)->None:
        """Update each weight by computing corresponding avg gradient and using formula: w(new) = w(old)-learningRate*avgGradient"""
        coeff = self.calculate_avg_gradients(self.gradients_holder_weights)

        for index, n in enumerate(self.weights):
            self.weights[index] = n-learning_rate*coeff[index]

        self.gradients_holder_weights=[]
        return
    
    def update_biase(self, learning_rate:float)->None:
        """Update bias by computing avg gradient and using formula: b(new) = b(old)-learningRate*avgGradient"""
        coeff = float(self.calculate_avg_gradients(self.gradients_holder_biases))

        self.bias = float(self.bias)-learning_rate*coeff
        # print(self.bias)
        self.gradients_holder_biases=[]
        return
    
    def get_bias(self)->float:
        """Getter for the bias"""
        return float(self.bias)

    def calculate_avg_gradients(self, gradients:list)->np.array:
        """Calculate average gradient for the each weight"""
        gradients_array = np.array(gradients)  
        return np.mean(gradients_array, axis=0)  
    
