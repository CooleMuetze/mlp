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
    output.set_y_true(y)
    print(f"Input: {output.get_input().flatten()} Output: {output.get_y_pred().flatten()} Expected: {output.get_y_true().flatten()} Loss: {output.get_loss()}")

for i in range(50000):
    for data in TrainingData:
        x = np.array([data[0], data[1]]).reshape(-1, 1)
        y = np.array(data[2]).reshape(-1, 1)
        output = mlp.forward(x)
        output.set_y_true(y)
        mlp.backpropagation(output, 0.1)


print("Post Training")
for data in TrainingData:
    x = np.array([data[0], data[1]]).reshape(-1, 1)
    y = np.array(data[2]).reshape(-1, 1)
    output = mlp.forward(x)
    output.set_y_true(y)
    print(f"Input: {output.get_input().flatten()} Output: {output.get_y_pred().flatten()} Expected: {output.get_y_true().flatten()} Loss: {output.get_loss()}")