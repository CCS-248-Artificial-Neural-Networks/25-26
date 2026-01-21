import numpy as np

class Dense_Layer:
    """
    Dense layer with activations and loss utilities.

    Take note:
    () Weights' shape should be (input_dimension, output_dimension)
    () Bias' shape should be (output_dimension,) or convertible to that shape
    () forward(inputs) expects inputs shape (input_dimension,) or (batch, input_dimension) for batch support (only 1D used here)
    
    () Supported activations:
        - ReLU (activation="relu")
        - Sigmoid (activation="sigmoid")
        - Softmax (activation="softmax")
        - Linear (activation="linear")

    () Use categorical_crossentropy(pred, target) for classification with softmax output where
        target is a probability distribution / one-hot vector (same length as output_dimension).
    """
    

    # (a.) A function to setup/accept the inputs and weights
    def __init__(self, weights, bias, activation="linear", name=None):
        """
        Initialization with validation of weights and bias.
        """
        try:
            self.weights = np.array(weights, dtype=float)
            self.bias = np.array(bias, dtype=float)
        except Exception as e:
            raise ValueError(f"[ERROR] Invalid weights/bias initialization: {e}")
        
        if self.bias.ndim == 0:
            self.bias = np.array([self.bias], dtype=float)
        
        if self.weights.ndim != 2:
            raise ValueError(f"[ERROR] Weights must be 2D array-like (input_dimension, output_dimension). Got shape {self.weights.shape}.")
        
        if self.bias.shape[-1] != self.weights.shape[1]:
            raise ValueError(f"[ERROR] Bias length {self.bias.shape} does not match weights output_dimension {self.weights.shape[1]}.")
        
        self.activation = activation.lower()
        self.name = name or f"Layer({self.activation})"

        # Print information for checking
        print(f"[INFO] Initialized {self.name}")
        print(f"       >> weights.shape = {self.weights.shape}, bias.shape = {self.bias.shape}")
        print(f"       >> expected input shape = ({self.weights.shape[0]},), output shape = ({self.weights.shape[1]},)\n")

    def forward(self, inputs):
        """
        Compute 'z' and activated output 'a' for given inputs.
        Expected output: (z, a) where
            z = weighted sum + bias
            a = activated output
        """

        print(f"[INFO] Forward pass through {self.name}")
        z = self._linear(inputs)
        a = self._activate(z)
        return z, a


    # (b.) A function to perform the weighted sum + bias
    def _linear(self, inputs):
        """
        Compute the linear combination (weighted sum + bias).
        Supports single input vector or batch of inputs.
        """
        inputs = np.array(inputs, dtype=float)
        
        if inputs.ndim == 1:
            if inputs.shape[0] != self.weights.shape[0]:
                raise ValueError(f"[ERROR] Input length {inputs.shape} does not match weights input_dimension {self.weights.shape[0]}.")
            z = np.dot(inputs, self.weights) + self.bias
        elif inputs.ndim == 2:
            if inputs.shape[1] != self.weights.shape[0]:
                raise ValueError(f"[ERROR] Input second dimension {inputs.shape[1]} does not match weights input_dimension {self.weights.shape[0]}.")
            z = np.dot(inputs, self.weights) + self.bias
        else:
            raise ValueError(f"[ERROR] Inputs must be 1D or 2D array-like. Got shape {inputs.shape}.")
        
        print(f"       >> Linear output (z): {z}")
        return z
    

    # (c.) A function to perform the selected activation function
    def _activate(self, z):
        """
        Apply the specified activation function to 'z'.
        """
        if self.activation == "relu":
            a = self._relu(z)
        elif self.activation == "sigmoid":
            a = self._sigmoid(z)
        elif self.activation == "softmax":
            a = self._softmax(z)
        elif self.activation == "linear":
            a = z
        else:
            raise ValueError(f"[ERROR] Unsupported activation '{self.activation}'. Supported: relu, sigmoid, softmax, linear.")
        
        print(f"       >> Activated output (a): {a}\n")
        return a
    
    # ----- Activation implementations -----
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def _softmax(self, x):
        x = np.array(x, dtype=float)
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    

    # (d.) A function to calculate the loss (predicted output vs target output)
    @staticmethod
    def crossentropy(predicted, target, eps=1e-12):
        """
        Computes cross-entropy loss.
        Automatically detects whether it's binary (sigmoid) or categorical (softmax).
        
        Parameters:
            predicted: predicted probabilities
            target: true label(s)
                - Binary: scalar/0D or shape (1,)
                - Categorical: one-hot array same length as predicted
        """

        predicted = np.array(predicted, dtype=float)
        target = np.array(target, dtype=float)

        predicted = np.clip(predicted, eps, 1.0 - eps)

        # ----- Case 1: Binary classification (1 output node) -----
        if predicted.shape == () or predicted.shape == (1,) or target.shape == () or target.shape == (1,):
            loss = -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))
            return float(loss)
        
        # ----- Case 2: Categorical classification (softmax outputs) -----
        elif predicted.ndim == 1 and target.shape == predicted.shape:
            loss = -np.sum(target * np.log(predicted))
            return float(loss)
        
        else:
            raise ValueError(
                f"[ERROR] Could not auto-detect loss type. "
                f"predicted.shape={predicted.shape}, target.shape={target.shape}"
            )


"""
Outline for the main code:
Let:
    - '#': float
    - '?': required string
    - '/': optional string
    - ...N: iteration


X = np.array([#, #, #, ...])
Target_output = np.array([#, #, #, ...])

W1 = [[#, #, #, ...], [...], ...]
B1 = [[#, #, #, ...], [...], ...]
Layer1 = Dense_Layer(W1, B1, activation="?", name="/")

W2 = [[#, #, #, ...], [...], ...]
B2 = [[#, #, #, ...], [...], ...]
Layer2 = Dense_Layer(W2, B2, activation="?", name="/")

WN = [[#, #, #, ...], [...], ...]
BN = [[#, #, #, ...], [...], ...]
LayerN = Dense_Layer(WN, BN, activation="?", name="/")
...

Z1, A1 = Layer1.forward(X)
Z2, A2 = Layer2.forward(A1)
Z3, A3 = Layer3.forward(A2)
...

loss = Dense_Layer.crossentropy(AN, Target_output)

print(?)

"""
