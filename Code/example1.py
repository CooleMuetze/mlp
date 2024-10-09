from MLP import MLP
from ActivatonFunction import *
import numpy as np

# Define the layers and activation functions
layers = [2, 2, 1]
activation = relu

# Create an MLP with the specified layers and activation functions
mlp = MLP(layers, activation)

# Input
x = np.array([2, 4]).reshape(-1, 1)

# Output (Zielwert entsprechend der ReLU-Ausgabe)
y = np.array([8])

# Forward pass (ausprobieren)
output = mlp.forward(x)
print("Initial output:", output[0][-1])
first = output[0][-1]
print("-----------------")

# Training loop
for i in range(1000):
    output = mlp.forward(x)
    print(f"Iteration {i}: {output[0][-1]}")
    mlp.backpropagation(output[0][-1], y, 0.01, output[0], output[1])  # Lernrate auf 0.1 setzen

# Final output
output = mlp.forward(x)
print("-----------------")
print("First answer: ", first)
print("Last answer: ", output[0][-1])