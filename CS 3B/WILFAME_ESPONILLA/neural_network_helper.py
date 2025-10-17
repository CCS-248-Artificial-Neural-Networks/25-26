# neural_network_helper.py
import math
import random

# ------------------------------
# a) Setup / Accept Inputs and Weights
# ------------------------------
def setup_inputs_and_weights(inputs, weights=None):
    """
    Accepts inputs and optionally custom weights.
    If no weights are provided, random weights are initialized.
    
    Parameters:
    inputs (list): Input values
    weights (list): Optional, initial weights
    
    Returns:
    tuple: (inputs, weights)
    """
    if weights is None:
        weights = [random.uniform(-1, 1) for _ in inputs]
    return inputs, weights


# ------------------------------
# b) Weighted Sum + Bias
# ------------------------------
def weighted_sum(inputs, weights, bias=0.0):
    """
    Performs weighted sum of inputs + bias.
    
    Parameters:
    inputs (list): Input values
    weights (list): Weight values
    bias (float): Bias term
    
    Returns:
    float: Weighted sum result
    """
    total = sum(i * w for i, w in zip(inputs, weights))
    return total + bias


# ------------------------------
# c) Activation Function
# ------------------------------
def activation_function(x, function_type="sigmoid"):
    """
    Applies the chosen activation function.
    
    Parameters:
    x (float): Input value
    function_type (str): 'sigmoid', 'relu', 'tanh'
    
    Returns:
    float: Activated output
    """
    if function_type == "sigmoid":
        return 1 / (1 + math.exp(-x))
    elif function_type == "relu":
        return max(0, x)
    elif function_type == "tanh":
        return math.tanh(x)
    else:
        raise ValueError("Unsupported activation function. Choose: sigmoid, relu, tanh.")


# ------------------------------
# d) Loss Function
# ------------------------------
def calculate_loss(y_pred, y_true, loss_type="mse"):
    """
    Calculates the loss between predicted and target output.
    
    Parameters:
    y_pred (float): Predicted output
    y_true (float): True output
    loss_type (str): 'mse' or 'cross_entropy'
    
    Returns:
    float: Loss value
    """
    if loss_type == "mse":
        return (y_true - y_pred) ** 2
    elif loss_type == "cross_entropy":
        # prevent log(0)
        epsilon = 1e-12
        y_pred = min(max(y_pred, epsilon), 1 - epsilon)
        return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))
    else:
        raise ValueError("Unsupported loss function. Choose: mse, cross_entropy.")
