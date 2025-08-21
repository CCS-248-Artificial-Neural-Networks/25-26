# Section 1: Pass/Fail Evaluation
print("=" * 40)
print("Number 1: Passed or Failed")
input_sets = [[8, 7], [3, 4]]
weights = [0.6, 0.4]
bias = -3

for set_num, values in enumerate(input_sets, 1):
    result = sum(val * w for val, w in zip(values, weights))
    print(f"\nSet {set_num} - Inputs: {values}")
    print("Output:", result)
    result_with_bias = result + bias
    print("Output with bias:", result_with_bias)
    is_active = result_with_bias > 1
    print("Activation:", is_active)
    print("-" * 30)

# Section 2: Logic Gate Emulation
print("=" * 40)
print("Number 2: Logic Gates Simulation | True or False")
input_sets = [[0, 0], [0, 1], [1, 0], [1, 1]]
weights = [1, 1]
bias = -1.5

for idx, pair in enumerate(input_sets, 1):
    gate_sum = sum(x * w for x, w in zip(pair, weights))
    print(f"\nSet {idx} - Inputs: {pair}")
    print("Output:", gate_sum)
    gate_sum_biased = gate_sum + bias
    print("Output with bias:", gate_sum_biased)
    gate_active = gate_sum_biased > 0
    print("Activation:", gate_active)
    print("-" * 30)

# Section 3: Perceptron Output Comparison
print("=" * 40)
print("Number 3: Perceptron Comparison")

input_vector = [0.5, -1, 2, 1, 0]
perceptron_weights = [
    [1.0, -0.5, 0.2, 0.1, 0.0],
    [0.2, 0.2, 0.5, -0.4, 0.3],
    [-0.3, -0.1, 0.4, 0.0, 0.2]
]
perceptron_biases = [0.2, 0.0, -0.6]
labels = ['A', 'B', 'C']
results = []

print(f"Input set: {input_vector}")
print("-" * 30)

for idx, (w_vec, b) in enumerate(zip(perceptron_weights, perceptron_biases)):
    net = sum(i * w for i, w in zip(input_vector, w_vec))
    tag = labels[idx]
    print(f"\nPerceptron {tag}:")
    print(f"Weights: {w_vec}, Bias: {b}")
    print("Output:", net)
    net_biased = net + b
    print("Output with bias:", net_biased)
    is_on = net_biased > 0
    print("Activation:", is_on)
    print("-" * 30)
    results.append((tag, net_biased))

# Announce Perceptron A's result
winner_output = results[0][1]
print(f"Winner of the comparison is Perceptron A with output: {winner_output}")