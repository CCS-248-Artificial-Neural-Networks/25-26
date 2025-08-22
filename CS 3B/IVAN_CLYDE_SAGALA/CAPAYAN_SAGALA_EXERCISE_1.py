# Problem 1: Student Pass/Fail Prediction


def perceptron_pass_fail(x1, x2):

    w1 = 0.6
    w2 = 0.4
    bias = -3
    threshold = 1

    total = (x1 * w1) + (x2 * w2) + bias

    if total >= threshold:
        return 1   # pass
    else:
        return 0   # fail

print("=== Problem 1: Student Pass/Fail ===")
print("Input (8,7) -> Output (Pass=1/Fail=0):", perceptron_pass_fail(8, 7))
print("Input (3,4) -> Output (Pass=1/Fail=0):", perceptron_pass_fail(3, 4))



# Problem 2: Logic Gate Simulation (AND Gate)


def perceptron_and(x1, x2):
    w1 = 1
    w2 = 1
    bias = -1.5
    threshold = 0

    total = (x1 * w1) + (x2 * w2) + bias

    if total >= threshold:
        return 1
    else:
        return 0

print("\n=== Problem 2: AND Gate Simulation ===")
inputs = [(0,0), (0,1), (1,0), (1,1)]
for x1, x2 in inputs:
    print("Input", (x1, x2), "-> Output:", perceptron_and(x1, x2))



# Problem 3: Perceptron Comparison (One-vs-All)


inputs = [0.5, -1, 2, 1, 0]

# Perceptron A
weights_A = [1.0, -0.5, 0.2, 0.1, 0.0]
bias_A = 0.2

# Perceptron B
weights_B = [0.2, 0.2, 0.5, -0.4, 0.3]
bias_B = 0.0

# Perceptron C
weights_C = [-0.3, -0.1, 0.4, 0.0, 0.2]
bias_C = -0.6

def perceptron(inputs, weights, bias):
    total = 0
    for i in range(len(inputs)):
        total = total + inputs[i] * weights[i]
    total = total + bias

    if total > 0:
        activation = 1
    else:
        activation = 0

    return activation, total

act_A, sum_A = perceptron(inputs, weights_A, bias_A)
act_B, sum_B = perceptron(inputs, weights_B, bias_B)
act_C, sum_C = perceptron(inputs, weights_C, bias_C)

print("\n=== Problem 3: Perceptron Comparison (One-vs-All) ===")
print("Perceptron A: Activation =", act_A, " Sum =", sum_A)
print("Perceptron B: Activation =", act_B, " Sum =", sum_B)
print("Perceptron C: Activation =", act_C, " Sum =", sum_C)

winner = "None"

if act_A == 1 or act_B == 1 or act_C == 1:
    winner = "A"
    max_sum = sum_A

    if act_B == 1 and sum_B > max_sum:
        winner = "B"
        max_sum = sum_B