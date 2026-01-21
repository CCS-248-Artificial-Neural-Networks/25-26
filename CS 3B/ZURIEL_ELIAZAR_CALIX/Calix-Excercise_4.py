import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# Generate dummy data (binary classification)
np.random.seed(42)
X = np.random.randn(200, 3)
y = np.random.randint(0, 2, 200).reshape(-1, 1)

# Training function for comparison
def train_model(optimizer_type, epochs=1000, lr=0.1, decay=0.001, momentum=0.9):
    # Initialize weights and bias
    weights = np.random.randn(3, 1) * 0.01
    bias = 0.0
    velocity = np.zeros_like(weights)
    cache = np.zeros_like(weights)
    losses = []
    accuracies = []
    prev_loss = None
    stable_epoch = None

    for epoch in range(epochs):
        # Learning rate decay
        lr_epoch = lr / (1 + decay * epoch)

        # Forward pass
        z = np.dot(X, weights) + bias
        output = sigmoid(z)
        error = y - output
        loss = np.mean((error) ** 2)
        losses.append(loss)

        # Accuracy
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions == y) * 100
        accuracies.append(accuracy)

        # Backpropagation
        d_output = error * sigmoid_derivative(output)
        grad_weights = np.dot(X.T, d_output) / X.shape[0]
        grad_bias = np.sum(d_output) / X.shape[0]

        # Optimizer logic
        if optimizer_type == "momentum":
            velocity = momentum * velocity + lr_epoch * grad_weights
            weights += velocity
        elif optimizer_type == "adaptive":
            cache = 0.9 * cache + 0.1 * (grad_weights ** 2)
            weights += lr_epoch * grad_weights / (np.sqrt(cache) + 1e-8)
        else:  # vanilla SGD
            weights += lr_epoch * grad_weights

        bias += lr_epoch * grad_bias

        # Detect stabilization (loss change < 0.001 for 10 consecutive epochs)
        if epoch > 10 and prev_loss is not None:
            if abs(prev_loss - loss) < 0.001 and stable_epoch is None:
                stable_epoch = epoch
        prev_loss = loss

    if stable_epoch is None:
        stable_epoch = epochs

    return stable_epoch, accuracies[-1], losses, accuracies

# Run both optimizers
stab_mom, acc_mom, losses_mom, accs_mom = train_model("momentum")
stab_adapt, acc_adapt, losses_adapt, accs_adapt = train_model("adaptive")

# Print summary table
print("\nComparison of Optimizers:")
print("Optimizer\tStabilization Epoch\tFinal Accuracy")
print(f"Momentum\t{stab_mom}\t\t\t{acc_mom:.2f}%")
print(f"Adaptive\t{stab_adapt}\t\t\t{acc_adapt:.2f}%")

# Optional: Plot loss curves for visual comparison (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    plt.plot(losses_mom, label="Momentum")
    plt.plot(losses_adapt, label="Adaptive")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.legend()
    plt.show()
except ImportError:
    pass
