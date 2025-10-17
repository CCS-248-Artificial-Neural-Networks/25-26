import numpy as np
import math

def setup_inputs_weights(inputs, weights):
    inputs = inputs
    weights = np.array(weights)
    return inputs, weights
        
def calculate_ws(inputs, weights, biases):
    return np.dot(inputs, weights) + biases
    
def calculate_softmax(layer_outputs):        
    exp_value = []
    for output in layer_outputs:
        exp_value.append(math.exp(output))
            
    norm_base = sum(exp_value)
    norm_values = []
    
    for value in exp_value:
        norm_values.append(value / norm_base)
        
    return norm_values

def calculate_relu(layer_outputs):
    relu_values = []
    
    for i in layer_outputs:
        relu_values.append(max(0, i))
        
    return relu_values

def calculate_sigmoid(layer_outputs):
    sigmoid_values = []
    
    for i in layer_outputs:
        sigmoid = 1 / (1 + math.exp(-i))
        sigmoid_values.append(sigmoid)
    
    return sigmoid_values
    
def calculate_loss(target_output, softmax_af):
    epsilon = 1e-15  # to avoid log(0)
    loss = 0
    for t, p in zip(target_output, softmax_af):
        loss -= t * math.log(p + epsilon)
        
    return loss