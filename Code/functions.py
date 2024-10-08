import numpy as np

### Activation functions

def sigmoid(x):
    """
    Sigmoid activation function
    :param x: input value
    :return: output of sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid activation function
    :param x: input value
    :return: output of the derivative of the sigmoid function
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """
    ReLU activation function
    :param x: input value
    :return: output of ReLU function
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of the ReLU activation function
    :param x: input value
    :return: output of the derivative of the ReLU function
    """
    return np.where(x > 0, 1, 0)

def noFunction(x):
    return np.array(x)

def noFunction_derivative(x):
    return np.array(1)