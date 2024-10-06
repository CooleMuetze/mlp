import numpy as np

class MLP:

    def __init__(self, layers: list, activation: list):
        """
        Initialize the MLP with the given layers and activation functions
        The length of the layers and activation functions should be the same
        :param list layers: list of integers, where each integer represents the number of neurons in that layer
        :param list activation: list of functions, where each function is the activation function for that layer
        :raises ValueError: if the length of the layers and activation functions are not the same
        """
        if len(layers) != len(activation):
            raise ValueError("The number of layers and activation functions should be the same, because each layer should have an activation function.")
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.biases = []
        self.initialize()

    def initialize(self):
        """
        Initialize the weights and biases for the MLP. Fills the weights 
        """
        for i in range(len(self.layers)):
            # Initialize random weights for the current layer i with a Matrix of shape (previous layer, current layer)
            # Typically set to a random value because it breaks symmetry and therefore sperates the neurons while training. This is important because if all the weights are the same, then all the neurons will learn the same thing.
            weight = np.random.randn(self.layers[i], self.layers[i - 1]) * 0.1

            # Initialize the bias for the current layer i with zeros
            # The bias is a constant value that is added to the activation function to shift the curve. It is initialized to zero because the activation function is centered around zero. 
            bias = np.zeros(self.layers[i])

            # Append the weights and biases to the list
            self.weights.append(weight)
            self.biases.append(bias)

    def forward_list(self, x: np.ndarray) -> list:
        """
        Forward pass through the MLP
        :param np.ndarray x: input to the MLP
        :return: complete list of results for each layer
        """

        # Define list of results for each layer. Represents the output of each layer used as input for the next layer. Stored as a list to visualize the flow of data through the network.
        results = []

        # Define the input as the first result
        results.append(x)

        # Loop through each layer
        for i in range(1, len(self.layers)):

            # Calculate the dot product of the input and the weights of the current layer
            # Add the bias to the dot product
            z = np.dot(self.weights[i], results[i-1]) + self.biases[i]

            # Apply the activation function to the result
            h = self.activation[i](z)

            # Append the result to the list of results
            results.append(h)

        # Return the output of the MLP
        return results
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the MLP
        :param np.ndarray x: input to the MLP
        :return: output of the MLP
        """
        # Loop through each layer
        for i in range(1, len(self.layers)):
            # Calculate the dot product of the input and the weights of the current layer
            # Add the bias to the dot product
            z = np.dot(self.weights[i], x) + self.biases[i]

            # Apply the activation function to the result
            x = self.activation[i](z)

        # Return the output of the MLP
        return x


    def sigmoid(x):
        """
        Sigmoid activation function
        :param x: input value
        :return: output of sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def relu(x):
        """
        ReLU activation function
        :param x: input value
        :return: output of ReLU function
        """
        return np.maximum(0, x)