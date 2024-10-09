from MLP import MLP
from ActivatonFunction import *
import numpy as np

layers = [2, 4, 1]

activation = sigmoid

mlp = MLP(layers, activation, biasIntialization=0.1)

TrainingData = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

print("Pre Training")
for data in TrainingData:
    x = np.array([data[0], data[1]]).reshape(-1, 1)
    y = np.array(data[2]).reshape(-1, 1)
    output = mlp.forward(x)
    print(f"Input: {x.flatten()} Output: {output[0][-1].flatten()} Expected: {y.flatten()}")

for i in range(50000):
    for data in TrainingData:
        x = np.array([data[0], data[1]]).reshape(-1, 1)
        y = np.array(data[2]).reshape(-1, 1)
        output = mlp.forward(x)
        mlp.backpropagation(output[0][-1], y, 0.1, output[0], output[1])


print("Post Training")
for data in TrainingData:
    x = np.array([data[0], data[1]]).reshape(-1, 1)
    y = np.array(data[2]).reshape(-1, 1)
    output = mlp.forward(x)
    print(f"Input: {x.flatten()} Output: {output[0][-1].flatten()} Expected: {y.flatten()}")