import numpy as np

class sigmoid:
    """
    This class defines the sigmoid activation function and its derivative
    """

    @staticmethod
    def run(x: np.ndarray) -> np.ndarray:
        """
        sigmoid activation function
        :param x: input value
        :return: output of sigmoid function
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function
        :param x: input value
        :return: output of the derivative of the sigmoid function
        """
        s = sigmoid.run(x)
        return s * (1 - s)

    
class relu:
    """
    This class defines the ReLU activation function and its derivative
    """

    @staticmethod
    def run(x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function
        :param x: input value
        :return: output of ReLU function
        """
        return np.maximum(0, x)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of the ReLU activation function
        :param x: input value
        :return: output of the derivative of the ReLU function
        """
        return np.where(x > 0, 1, 0)
    

class noFunction:
    """
    This class defines the linear activation function and its derivative used for no activation function
    """

    @staticmethod
    def run(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.array(1)
