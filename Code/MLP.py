import numpy as np

class MLP:

    def __init__(self, layers: list, activation, activation_derivative):
        """
        Initialize the MLP with the given layers and activation functions
        The length of the layers and activation functions should be the same
        :param list layers: list of integers, where each integer represents the number of neurons in that layer
        :param list activation: list of functions, where each function is the activation function for that layer
        """
        self.layers = layers
        self.activation = activation
        self.activation_derivative = activation_derivative
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
            # Use np.reshape to make sure that the bias is a column vector
            bias = np.reshape(np.zeros(self.layers[i]), (-1, 1))

            # Append the weights and biases to the list

            self.weights.append(weight)
            self.biases.append(bias)

        


    ### Forward pass


    def forward(self, x: np.ndarray):
        """
        Forward pass through the MLP
        :param np.ndarray x: input to the MLP
        :return: complete list of results for each layer
        """

        # Define list of results for each layer. Represents the output of each layer used as input for the next layer. Stored as a list to visualize the flow of data through the network.
        h_list = []
        z_list = []

        # Define the input as the first result
        h_list.append(x.reshape(-1, 1))
        z_list.append(x.reshape(-1, 1))


        # Loop through each layer
        for i in range(1, len(self.layers)):

            # Calculate the dot product of the input and the weights of the current layer
            # Add the bias to the dot product
            z = np.dot(self.weights[i], h_list[i-1]) + self.biases[i]

            z_list.append(z)

            # Apply the activation function to the result
            h = self.activation(z)

            h_list.append(h)

        # Return the output of the MLP
        return h_list, z_list


    ### Help funktion for Backpropagation

    def mse(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Mean Squared Error (MSE) loss function
        :param np.ndarray y_pred: predicted output
        :param np.ndarray y_true: target output
        :return: MSE loss
        """
        sum = 0
        for i in range(len(y_pred)):
            sum += (y_pred[i] - y_true[i]) ** 2
        return sum / y_pred.size
    
    def mse_derivative(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Derivative of the Mean Squared Error (MSE) loss function
        :param np.ndarray y_pred: predicted output
        :param np.ndarray y_true: target output
        :return: derivative of the MSE loss
        """
        return 2 * (y_pred - y_true) / y_true.size



    ### Backpropagation

    def backpropagation(self, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float, h_list: list, z_list: list):
        """
        Backpropagation algorithm to train the MLP
        :param np.ndarray y_pred: predicted output
        :param np.ndarray y_true: target output
        :param float learning_rate: learning rate
        :param list h_list: list of results for each layer
        :param list z_list: list of results for each layer before activation
        """

        delta = [None] * len(self.layers)
        delta[len(self.layers) - 1] = (self.mse_derivative(y_pred, y_true).reshape(-1, 1) * self.activation_derivative(z_list[len(self.layers) - 1]).reshape(-1, 1))

        for l in range(len(self.layers) - 1, 0, -1):

            delta[l - 1] = (np.dot(self.weights[l].T, delta[l]) * self.activation_derivative(z_list[l - 1]))
            
        
        for l in range(1, len(self.layers)):
            
            self.weights[l] -= learning_rate * np.dot(delta[l], (h_list[l - 1]).T)
            self.biases[l] -= learning_rate * delta[l]
            
