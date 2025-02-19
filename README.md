# Multilayer Perceptron for Handwritten Digit Recognition
Inspired by Neural Networks and Deep Learning by Michael Nielsen.(http://neuralnetworksanddeeplearning.com)

## Overview
This project implements one of the neuronal networks –  **Multilayer Perceptron (MLP)** for **handwritten digit recognition**, based on concepts from Neural Networks and Deep Learning by Michael Nielsen. The model is trained on the **MNIST dataset**, which consists of handwritten digits from 0 to 9.

## Requirements
Make sure you have the following dependencies installed:

* Python 3.x
* ```numpy``` (for numerical computations)

## Installation
Clone the repository:
```sh
git clone https://github.com/t3i8m/Multilayer-Perceptron.git
cd Multilayer-Perceptron
```
Install dependencies:
```sh
pip install numpy 
```

## Usage
Run the program using main.py:
```sh
python main.py
```
## Dataset
This project uses the **MNIST dataset**, a collection of 60,000 training images and 10,000 test images of handwritten digits (0-9).
Each image is **28x28** pixels in grayscale.

## Structure 
```sh

Multilayer-Perceptron/
│── __pycache__/             # Compiled Python files
│── data/                    # Dataset storage
│   ├── mnist.pkl.gz         # MNIST dataset in pickle format
│── nn/                      # Neural network (scratch implementation)
│   ├── __pycache__/         # Compiled Python files
│   ├── MLP_scratch.py       # MLP implementation from scratch
│── nn_oop/                  # Object-Oriented Neural Network implementation
│   ├── __pycache__/         # Compiled Python files
│   ├── __init__.py          # Package initialization
│   ├── layer.py             # Layer implementation
│   ├── MLP.py               # Main MLP class
│   ├── neuron.py            # Neuron implementation
│── utils/                   # Utility functions
│   ├── __pycache__/         # Compiled Python files
│   ├── __init__.py          # Package initialization
│   ├── activation_loss.py   # Activation functions and loss calculations
│   ├── mnist_loader.py      # MNIST dataset loader
│── main.py                  # Main script to run the model
│── README.md                # Project documentation
```
## License
MIT License