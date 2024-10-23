
# train mnist model
from ActivatonFunction import *
import numpy as np
import pickle
from keras.datasets import mnist
from MLP import MLP
from datetime import datetime as Date

# load data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28*28)

# Define the layers and activation functions
layers = [28*28, 128, 10]
activation = sigmoid

# Create an MLP with the specified layers and activation functions
mlp = MLP(layers, activation)


# Load model to finetune comment out if you want to train a new model
with open('example/mnist/mnist.pkl', 'rb') as file:
    mlp = pickle.load(file)


# Training loop
loops = 20
for i in range(loops):
    starttime = Date.now()
    print(f"--- {i+1}/{loops}")
    print("Starting Iteration")
    starttime = Date.now()
    for j in range(len(x_train)):
        x = x_train[j].reshape(-1, 1)
        y = np.zeros((10, 1))
        y[y_train[j]] = 1
        output = mlp.forward(x)
        output.set_y_true(y)
        mlp.backpropagation(output, learning_rate=0.002)
        

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
    
    print(f"Accuracy after loop: {correct/len(x_test)}")
    # endtime in seconds formated as int
    endtime = Date.now()
    # save time difference in seconds as int
    duration = (endtime-starttime).seconds
    print(f"Time for Iteration {i+1}/{loops}: {duration}s")
    remainingloops = loops-i-1
    remainingTime = remainingloops*(duration)
    print(f"Estimated Time for remaining {remainingloops} loops: {remainingTime}s or {remainingTime/60}min")

#save model
with open('example/mnist/mnist.pkl', 'wb') as file:
    pickle.dump(mlp, file)