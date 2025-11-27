import numpy as np 
import math
#a) a function that initializes the weights and inputs 
def setup_layer(inputs, weights):
    inputs = np.array(inputs)
    weights = np.array(weights)
    return inputs, weights

#b) A function to perform the weighted sum + bias
def weighted_sum(inputs, weights, biases):
    inputs, weights = setup_layer(inputs, weights)
    biases = np.array(biases)
    return np.dot(inputs, weights.T) + biases

#c) A function to perform the selected activation function
def activation_function(x, activation_type="relu"):
    if activation_type == "relu":
        return np.maximum(0, x)
    elif activation_type == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation_type == "softmax":
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    else:
        raise ValueError("Unsupported activation type")

#d) A function that calculates the loss (categorical Cross Entropy loss) (predicted output vs actual output)
def calculate_loss(target, actual, loss_type="softmax"):
    target = np.array(target)
    actual = np.array(actual)
    epsilon = 1e-12
    actual = np.clip(actual, epsilon, 1 - epsilon)  # avoid log(0)
    
    if loss_type == "softmax":
        # categorical cross-entropy
        loss = -np.mean(np.sum(target * np.log(actual), axis=-1))
        return loss

    elif loss_type == "sigmoid":
        # binary cross-entropy
        loss = -np.mean(target * np.log(actual) + (1 - target) * np.log(1 - actual))
        return loss

    else:
        raise ValueError("Invalid loss_type. Choose 'softmax' or 'sigmoid'.")
   