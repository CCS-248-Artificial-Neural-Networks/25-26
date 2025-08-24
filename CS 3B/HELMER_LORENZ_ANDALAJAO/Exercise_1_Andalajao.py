# Problem 1: Student Pass/Fail
def predict_pass_fail(hours_studied, hours_sleep, hours_free):
    w1 = 0.6
    w2 = 0.4
    w3 = 0.0
    bias = -3
    threshold = 1
    weighted_sum = (hours_studied * w1) + (hours_sleep * w2) + (hours_free * w3) + bias
    if weighted_sum >= threshold:
        return 1
    else:
        return 0

print("=== Problem 1: Student Pass/Fail ===")
print("Input (8,7,0) -> Output:", predict_pass_fail(8, 7, 0))
print("Input (3,4,0) -> Output:", predict_pass_fail(3, 4, 0))


# Problem 2: Logic Gate (AND)
def perceptron_and_gate(x1, x2):
    w1 = 1
    w2 = 1
    bias = -1.5
    threshold = 0
    weighted_sum = (x1 * w1) + (x2 * w2) + bias
    if weighted_sum >= threshold:
        return 1
    else:
        return 0

print("\n=== Problem 2: AND Gate Simulation ===")
and_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
for x1, x2 in and_inputs:
    print("Input", (x1, x2), "-> Output:", perceptron_and_gate(x1, x2))


# Problem 3: Perceptron Comparison
input_values = [0.5, -1, 2, 1, 0]

weights_a = [1.0, -0.5, 0.2, 0.1, 0.0]
bias_a = 0.2

weights_b = [0.2, 0.2, 0.5, -0.4, 0.3]
bias_b = 0.0

weights_c = [-0.3, -0.1, 0.4, 0.0, 0.2]
bias_c = -0.6

def perceptron_activation(inputs, weights, bias):
    weighted_sum = sum(inputs[i] * weights[i] for i in range(len(inputs))) + bias
    activation = 1 if weighted_sum > 0 else 0
    return activation, weighted_sum

activation_a, sum_a = perceptron_activation(input_values, weights_a, bias_a)
activation_b, sum_b = perceptron_activation(input_values, weights_b, bias_b)
activation_c, sum_c = perceptron_activation(input_values, weights_c, bias_c)

print("\n=== Problem 3: Perceptron Comparison (One-vs-All) ===")
print("Perceptron A: Activation =", activation_a, " Sum =", sum_a)
print("Perceptron B: Activation =", activation_b, " Sum =", sum_b)
print("Perceptron C: Activation =", activation_c, " Sum =", sum_c)

winner = "None"
max_sum = float("-inf")

if activation_a == 1 and sum_a > max_sum:
    winner = "Perceptron A"
    max_sum = sum_a
if activation_b == 1 and sum_b > max_sum:
    winner = "Perceptron B"
    max_sum = sum_b
if activation_c == 1 and sum_c > max_sum:
    winner = "Perceptron C"
    max_sum = sum_c

print("Winner:", winner)
