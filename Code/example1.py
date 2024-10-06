from MLP import MLP
import numpy as np

# Define the layers and activation functions
layers = [2, 3, 5]
activation = [None, MLP.sigmoid, MLP.sigmoid]

# Create an MLP with the specified layers and activation functions
mlp = MLP(layers, activation)

# Define the input to the MLP
x = np.array([0.1, 5])

# Perform a forward pass through the
# MLP and print the output
output = mlp.forward_list(x)
print(output)