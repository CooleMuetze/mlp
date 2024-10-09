
class forwardResult:

    def __init__(self, h_list, z_list):
        """
        Initialize the forward result
        :param h_list: list of results for each layer
        :param z_list: list of results for each layer before activation function applied
        """
        self.h_list = h_list
        self.z_list = z_list

    def set_y_true(self, y_true):
        """
        Set the expected output
        :param y_true: expected output
        """
        self.y_true = y_true

    def get_y_true(self):
        """
        Get the expected output
        :return: expected output
        """
        return self.y_true
    
    def get_y_pred(self):
        """
        Get the predicted output
        :return: predicted output
        """
        return self.h_list[-1]

    def get_h_list(self):
        """
        Get the list of results for each layer
        :return: list of results for each layer
        """
        return self.h_list
    
    def get_z_list(self):
        """
        Get the list of results for each layer before activation function applied
        :return: list of results for each layer before activation function applied
        """
        return self.z_list
    
    def get_input(self):
        """
        Get the input vector to the MLP
        :return: input to the MLP
        """
        return self.h_list[0]
    
    def get_loss(self):
        """
        Get the mse loss of the forward pass
        :return: mse loss of the forward pass
        """
        sum = 0
        for i in range(len(self.get_y_pred())):
            sum += (self.get_y_pred()[i] - self.y_true[i]) ** 2
        return sum / self.get_y_pred().size