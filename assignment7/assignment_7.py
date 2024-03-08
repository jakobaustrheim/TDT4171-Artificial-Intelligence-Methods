import numpy as np


amount_of_weights = 2
layers = 3
learning_rate = 0.1

def sigmoid(X: float):
    return 1/(1+np.exp(-X))

def d_sigmoid(X: float):
    return np.exp(-X)/((np.exp(-X)+1)**2)


class Node(): #Creating a class that represetns a single node in the network
    def __init__(self, amount_of_weights: int = 2):
        self.a = 0
        self.weights = np.random.rand(amount_of_weights) * 2-1


class network(): #Creating a class for the network that contains
    def __init__(self):
        amount_of_nodes = 2

        self.output_layer = [Node(0)]
        self.first_layer = []
        self.hidden_layer = []

        for i in range(amount_of_nodes):
            self.first_layer[i] = Node()
            self.hidden_layer[i] = Node()
    
""""
I have created a network. Now i need functions for forwaring through the network, and then for back-propagating to shift the weights
"""        

def forwarding() -> Node:
    


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # TODO: Your code goes here.

class Node():

    def __init__(self):

        pass


class NeuralNetwork():

    def __init__(self):
        #input dim
        #output dim
        #hidden_dim
        pass

    def forward(self):

        pass

    def backward(self):

        pass

    def train(self):

        pass

    def test(self):

        pass

