import pickle
from keras.datasets import mnist
import numpy as np

print("Start Testing the model")

with open('example/mnist/mnist.pkl', 'rb') as file:
    mlp = pickle.load(file)

# load data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28*28)

# Test model accuracy
correct = 0
for j in range(len(x_test)):
    x = x_test[j].reshape(-1, 1)
    output = mlp.forward(x)
    y = np.zeros((10, 1))
    y[y_test[j]] = 1
    output.set_y_true(y)
    if output.is_correct():
        correct += 1
print(f"Accuracy for Test Data: {correct/len(x_test)}")

# Test model accuracy with Training Data
correct = 0
for j in range(len(x_train)):
    x = x_train[j].reshape(-1, 1)
    output = mlp.forward(x)
    y = np.zeros((10, 1))
    y[y_train[j]] = 1
    output.set_y_true(y)
    if output.is_correct():
        correct += 1
print(f"Accuracy for Train Data: {correct/len(x_train)}")