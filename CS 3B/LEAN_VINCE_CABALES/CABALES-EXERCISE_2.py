import numpy as np
from neural_network_helper import Dense_Layer

"""
b)	Given the following inputs from the Breast Cancer Dataset, 
using three features: Mean Radius, Mean Texture, and Mean Smoothness, 
determine whether the tumor is Benign (0) or Malignant (1) by calculating 
the network outputs step by step, given the following neural network configuration.
"""

# ---------- Problem 2-b: Breast Cancer dataset ----------

# Inputs: (Mean Radius, Mean Texture, Mean Smoothness)
X = np.array([14.1, 20.3, 0.095])
Target_output = 1 # malignant

# First Hidden Layer
W1 = [
    [0.5, 0.2, -0.7],
    [-0.3, 0.4, 0.9],
    [0.8, -0.6, 0.1]
]
B1 = [0.3, -0.5, 0.6]
Layer1 = Dense_Layer(W1, B1, activation="relu", name="Hidden1-ReLU")

# Second Hidden Layer
W2 = [
    [0.6, -0.3],
    [-0.2, 0.5],
    [0.4, 0.7]
]
B2 = [0.1, -0.8]
Layer2 = Dense_Layer(W2, B2, activation="sigmoid", name="Hidden2-Sigmoid")

# Third Hidden Layer
W3 = [
    [0.7],
    [-0.5]
]
B3 = [0.2]
Layer3 = Dense_Layer(W3, B3, activation="sigmoid", name="Hidden3-Sigmoid")

# Forward Propagation
Z1, A1 = Layer1.forward(X)
Z2, A2 = Layer2.forward(A1)
Z3, A3 = Layer3.forward(A2)

# Loss
loss = Dense_Layer.crossentropy(A3, Target_output)

# Summary Print
print("\n=== Summary (Problem 2-b: Breast Cancer) ===")
print("Input X:", X)
print("Hidden Layer 1 - z1:", np.round(Z1, 6))
print("Hidden Layer 1 - a1 (ReLU):", np.round(A1, 6))
print("Hidden Layer 2 - z2:", np.round(Z2, 6))
print("Hidden Layer 2 - a2 (Sigmoid):", np.round(A2, 6))
print("Output Layer - z3:", np.round(Z3, 6))
print("Output Layer - a3 (Sigmoid prediction):", np.round(A3, 6))
print("Target Output:", Target_output)
print("Binary Cross-Entropy Loss:", float(np.round(loss, 6)))

# Conclusion Print
A3_scalar = float(np.squeeze(A3))
predicted_label = int(A3_scalar >= 0.5)
label_str = "Malignant (1)" if predicted_label == 1 else "Benign (0)"
print("\n=== Conclusion ===")
print(f"The network prediction: {label_str}")
print(f"Predicted probability of being malignant: {A3_scalar:.6f}")
