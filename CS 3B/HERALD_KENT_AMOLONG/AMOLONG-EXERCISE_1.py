# Number 1: Passed or Failed
print("=" * 40)
print("Number 1: Passed or Failed")
input_sets = [[8,7], [3,4]] 
weights = [0.6, 0.4]
bias = -3

for idx, inputs in enumerate(input_sets):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    
    print(f"\nSet {idx+1} - Inputs: {inputs}")
    print("Output:", output)
    
    output += bias
    print("Output with bias:", output)
    
    active = output > 1
    print("Activation:", active)
    print("-" * 30)

# Number 2: Logic Gates Simulation| True or False
print("=" * 40)
print("Number 2: Logic Gates Simulation | True or False")
input_sets = [[0,0], [0,1], [1,0], [1,1]]
weights = [1, 1]
bias = -1.5

for idx, inputs in enumerate(input_sets):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    
    print(f"\nSet {idx+1} - Inputs: {inputs}")
    print("Output:", output)
    
    output += bias
    print("Output with bias:", output)
    
    active = output > 0
    print("Activation:", active)
    print("-" * 30)

# Number 3: Perceptron Comparison
print("=" * 40)
print("Number 3: Perceptron Comparison")

inputs = [0.5,-1,2,1,0] 
weight_sets = [[1.0,-0.5,0.2,0.1,0.0], [0.2,0.2,0.5,-0.4,0.3], [-0.3,-0.1,0.4,0.0,0.2]] 
biases = [0.2, 0.0, -0.6]
perceptron_names = ['A', 'B', 'C']
perceptron_outputs = []

print(f"Input set: {inputs}")
print("-" * 30)

for idx, (weights, bias) in enumerate(zip(weight_sets, biases)):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    
    name = perceptron_names[idx]
    print(f"\nPerceptron {name}:")
    print(f"Weights: {weights}, Bias: {bias}")
    print("Output:", output)
    
    output += bias
    print("Output with bias:", output)
    
    active = output > 0
    print("Activation:", active)
    print("-" * 30)
    
    perceptron_outputs.append((name, output))

# Store Perceptron A's output for the final message
a_output = perceptron_outputs[0][1]
print(f"Winner of the comparison is Perceptron A with output: {a_output}")