from nn_components.MLP import NeuralNetwork
from mnist_loader import load_data_wrapper

def main():
    net = NeuralNetwork(2, [16,16])
    training_data , validation_data , test_data = load_data_wrapper()
    print("------------Data was loaded------------------")
    net.SGD(training_data , 30, 10, 3.0, test_data = test_data )


if __name__ == "__main__":
    main()