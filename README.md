# Understanding MLP and Backward Propagation

## Introduction
This project is designed to help you understand the mathematics behind Multi-Layer Perceptrons (MLP) and Backward Propagation. It is not intended to be an efficient implementation of these concepts. The primary goal is to provide a clear and educational explanation of the underlying mathematical principles.

## Multi-Layer Perceptron (MLP)
A Multi-Layer Perceptron is a type of artificial neural network that consists of multiple layers of nodes. Each node, or neuron, in one layer is connected to every node in the next layer. MLPs are used for various tasks such as classification and regression.

### Basic Concepts:
- **Input Layer**: The layer that receives the input data.
- **Hidden Layers**: Layers between the input and output layers where computations are performed.
- **Output Layer**: The layer that produces the final output.

## Backward Propagation
Backward Propagation is a method used to train neural networks. It involves calculating the gradient of the loss function with respect to each weight by the chain rule, and then updating the weights to minimize the loss.

### Key Steps:
1. **Forward Pass**: Compute the output of the network.
2. **Loss Calculation**: Calculate the error between the predicted output and the actual output.
3. **Backward Pass**: Compute the gradient of the loss function with respect to each weight.
4. **Update**: Adjust the weights and biases to minimize the loss.

## Code Structure
The implementation of MLP and Backward Propagation can be found in the `MLP.py` file. The code is structured to clearly illustrate each step of the process. Additionally, the files `forwardResult.py` and `ActivationFunction.py` are helper files designed to simplify the usage of an MLP without requiring detailed knowledge of each vector. These helper files make it easier to work with the MLP class by abstracting some of the complexities involved.

## Examples
You can find various examples in the `examples` directory. These examples demonstrate different use cases and help you understand how to implement and utilize the MLP and Backward Propagation concepts in practice. To execute an example, use the `-m` parameter. For example for example 2:
```bash
python -m example.simple.example2
```
Please ensure you are in the root directory of the project when running the examples.

## MNIST Example
The project includes an example using the MNIST dataset, where you can draw your own digits and see how well the model recognizes them.
> **Note:** The recognition may not be optimal, as MNIST is designed for a specific style of handwriting that may not match your own.

The uploaded model achieves an accuracy of 92% on the test datasets.

## Disclaimer
The primary objective of this project is to understand the mathematics behind MLP and Backward Propagation. It is not optimized for performance or accuracy.

## Conclusion
I hope this project helps you gain a deeper understanding of the mathematical concepts behind neural networks.

## About the Author
This project was created by me, Leonard Grün. While working on this project, I am studying Business Informatics at the Karlsruhe Institute of Technology (KIT).
