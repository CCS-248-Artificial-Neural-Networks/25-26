# 1. a.	Develop a Class in Python called Dense_Layer (included in the submitted notebook).

#a)	(10 points) A function to setup/accept the inputs and weights
#b)	(10 points) A function to perform the weighted sum + bias
#c)	(15 points) A function to perform the selected activation function
#d)	(15 points) A function to calculate the loss (predicted output vs target output)

import numpy as np

class Dense_Layer:

    def __init__(self): #setup/accept inputs and weights
        
        self.inputs = None
        self.weights = None
        self.bias = None
        self.output = None
        

    def summation(self, inputs, weights, bias): #perform weighted sum + bias

        self.inputs = np.array(inputs)
        self.weights =np.array(weights)
        self.bias = np.array(bias)
        
        z = np.dot(self.inputs, self.weights) + self.bias
        return z

    
    def activation(self, z, func="relu"): #for activation function

        if func == "relu":
            self.output = np.maximum(0,z)
        elif func == "sigmoid":
            self.output = 1/(1 + np.exp(-z))
        elif func == "softmax":
            exp_values = np.exp(z - np.max(z, axis = -1, keepdims = True))
            self.output = exp_values / np.sum(exp_values, axis = -1, keepdims = True)
        else:
            raise ValueError("Unsupported activation function.")
        return self.output

    def loss_function(self, y_target): # for loss function

        y_target = np.array(y_target)
        y_pred = self.output

        lf = 0.5 * np.sum((y_target - y_pred) ** 2)
        return lf