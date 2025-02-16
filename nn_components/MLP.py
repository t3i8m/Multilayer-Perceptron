import numpy as np
from nn_components.layer import Layer
from utils import loss_function, sigma_prime_from_a, sigmoid_function
import random

class NeuralNetwork(object):

    def __init__(self, num_layers:int, num_neurons:list):
        """Constructor for the MLP object"""
        # valid checks
        if(num_layers!=len(num_neurons)):
            raise ValueError("Mismath in the number of layers and num_neurons length. Must be the same!")
        
        if(type(num_layers)!=int and type(num_neurons)!=list):
            raise TypeError("Mismath in the arguments types!")

        # add neurons` number for the input/output layers  
        num_neurons = [784]+num_neurons+[10]

        # set up layers
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.params = [prev_neuron_num*neuron_number+neuron_number for prev_neuron_num, neuron_number in zip(num_neurons[:-1], num_neurons[1:])]

        self.input_layer = Layer(784, 0)
        self.input_layer.set_activations(np.zeros(784))

        self.output_layer = Layer(10, num_neurons[-2])
        self.layers = [self.input_layer]
        self.layers.extend([Layer(current_neuron_num, prev_layer) for current_neuron_num, prev_layer in zip(num_neurons[1:num_layers+1], num_neurons[:num_layers])])
        self.layers.append(self.output_layer)

    def feedforward(self, a:np.array)->np.array:
        """Return the output of the network if "a" is input."""
        # check for the "a" vector shape and space
        a = a.reshape(-1, 1) if a.ndim == 1 else a

        for layer in self.layers[1:]:
            layer_weights = layer.get_weights()
            layer_biases = layer.get_biases()
            a = sigmoid_function(np.dot(layer_weights, a)+layer_biases)
            layer.set_activations(a)
        return a

    def stochastic_gradient_descent(self, training_data:list, epochs:int, mini_batch_size:int, learning_rate:float, test_data=None)->None:
        """Train the neural network using mini-batch stochastic gradient descent. The
        "training_data" is a list of tuples "(x, y)" representing the training
        inputs and the desired outputs. The other non-optional parameters are self -
        explanatory. If "test_data" is provided then the network will be evaluated
        against the test data after each epoch , and partial progress printed out.
        This is useful for tracking progress , but slows things down substantially.
        """
        if test_data:
            test_size = len(test_data)

        for n in range(epochs):
            # shuffle the training data before each epoch
            random.shuffle(training_data)

            mini_batches = [training_data[n:mini_batch_size] for n in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batches(mini_batch, learning_rate)

            if(test_data):
                print(f"Epoch {n}: {self.evaluate(test_data)} / {test_size}")
            else:
                print(f"Epoch {n} complete")
        return

    def update_mini_batches(self, mini_batch:list, learning_rate:float):
        """Update the network’s weights and biases by applying gradient descent using
        backpropagation to a single mini batch. The "mini_batch" is a list of tuples
        "(x, y)", and the learning rate."""

        gradients = [[np.zeros(n)] for n in self.params]

        for index, n in enumerate(mini_batch):

            self.backprop()




        return

    def backprop(self, batch:tuple):
        """Use a backpropogation algorithm to calculate gradients for each weight and bias."""

        # calculate gradients in the output layer
        output_layer = self.layers[-1]
        error = 2*(output_layer.get_activations()-batch[1])*sigma_prime_from_a(output_layer.get_activations())
        gradients_output = np.outer(error, self.layers[-2].get_activations())
        output_layer.set_gradients_weights(gradients_output)
        output_layer.set_gradients_bias(error)

        error_next = error
        for index in range(len(self.layers)-2, 0, -1):

            layer = self.layers[index]
            next_layer = self.layers[index + 1]
            weights_next = next_layer.get_weights() 

            error = np.dot(weights_next.T, error_next)*sigma_prime_from_a(layer.get_activations())
            activations_prev = self.layers[index - 1].get_activations()
            gradients = np.outer(error, activations_prev)

            layer.set_gradients_weights(gradients)
            layer.set_gradients_bias(error)

            error_next = error

        return


    