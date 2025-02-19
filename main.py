from nn.MLP_scratch import NeuralNetworkRaw
from utils.mnist_loader import load_data_wrapper

def main():
    net = NeuralNetworkRaw([784, 30, 10])
    training_data , validation_data , test_data = load_data_wrapper()

    print("------------Data was loaded------------------")
    net.SGD(training_data , 30, 32, 3.0, test_data = test_data )


if __name__ == "__main__":
    main()