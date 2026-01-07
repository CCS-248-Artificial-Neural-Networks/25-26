import numpy as np
from neural_network_helper import Dense_Layer

# Input and Target 
X = np.array([5.1, 3.5, 1.4, 0.2])
Target_output = np.array([0.7, 0.2, 0.1])  

# First Hidden Layer
W1 = [
    [0.2, 0.5, -0.3],
    [0.1, -0.2, 0.4],
    [-0.4, 0.3, 0.2],
    [0.6, -0.1, 0.5]
]
B1 = [3.0, -2.1, 0.6]
Layer1 = Dense_Layer(W1, B1, activation="relu", name="Hidden1-ReLU")

# Second Hidden Layer 
W2 = [
    [0.3, -0.5],
    [0.7, 0.2],
    [-0.6, 0.4]
]
B2 = [4.3, 6.4]
Layer2 = Dense_Layer(W2, B2, activation="sigmoid", name="Hidden2-Sigmoid")

# Last Hidden Layer
W3 = [
    [0.5, -0.3, 0.8],
    [-0.2, 0.6, -0.4]
]
B3 = [-1.5, 2.1, -3.3]
Layer3 = Dense_Layer(W3, B3, activation="softmax", name="Output-Softmax")

Z1, A1 = Layer1.forward(X)
Z2, A2 = Layer2.forward(A1)
Z3, A3 = Layer3.forward(A2)

# Loss
loss = Dense_Layer.crossentropy(A3, Target_output)

# Summary 
print("\n Summary")
print("Input X:", X)
print("Hidden Layer 1 - z1:", np.round(Z1, 6))
print("Hidden Layer 1 - a1 (ReLU):", np.round(A1, 6))
print("Hidden Layer 2 - z2:", np.round(Z2, 6))
print("Hidden Layer 2 - a2 (Sigmoid):", np.round(A2, 6))
print("Last Layer - z3:", np.round(Z3, 6))
print("Last Layer - a3 (Softmax):", np.round(A3, 6))
print("Target Output:", Target_output)
print("Loss:", float(np.round(loss, 6)))

predicted_class = int(np.argmax(A3))
class_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
print("\n Conclusion ")
print(f"The network prediction: {class_labels[predicted_class]}")
print(f"Predicted probability distribution: {A3}")