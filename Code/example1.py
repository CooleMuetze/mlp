from MLP import MLP
from functions import *
import numpy as np

# Define the layers and activation functions
layers = [2, 2, 1]
activation = noFunction
activation_derivative = noFunction_derivative

# Create an MLP with the specified layers and activation functions
mlp = MLP(layers, activation, activation_derivative)

# Input
x = np.array([2, 4])

# Output
y = np.array([8])

# Perform a forward pass through the
# MLP and print the output

output = mlp.forward(x)
print(output)
print(output[1][len(layers)-1])
first = output[1][len(layers)-1]
print("-----------------")



for i in range(150):
    output = mlp.forward(x)
    print(output[1][len(layers)-1])
    mlp.backpropagation(output[0][-1:], y, 0.001, output[0], output[1])


output = mlp.forward(x)
print("-----------------")
print("First answer: ", first)
print("Last answer: ", output[1][len(layers)-1])

