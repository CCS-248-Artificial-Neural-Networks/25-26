
# Problem 1: Student Pass/Fail
print("=== Problem 1: Student Pass/Fail ===")

w1 = 0.6
w2 = 0.4
bias = -3
threshold = 1

test_inputs = [(8, 7), (3, 4)]

for x1, x2 in test_inputs:
    output = 0
    output += x1 * w1
    output += x2 * w2
    print(f"Input ({x1}, {x2}) → Weighted Sum (before bias): {output:.2f}")

    output += bias
    print(f"  With bias: {output:.2f}")

    activate = output >= threshold
    print(f"  Activation: {activate} → {'Pass (1)' if activate else 'Fail (0)'}\n")



# Problem 2: Logic Gate (AND)
print("=== Problem 2: Logic Gate (AND Simulation) ===")

# Given perceptron setup
w1 = 1
w2 = 1
bias = -1.5
threshold = 0

# All input combinations
gate_inputs = [(0,0), (0,1), (1,0), (1,1)]

for x1, x2 in gate_inputs:
    output = 0
    output += x1 * w1
    output += x2 * w2
    print(f"Input ({x1}, {x2}) → Weighted Sum (before bias): {output:.2f}")

    output += bias
    print(f"  With bias: {output:.2f}")

    activate = output >= threshold
    print(f"  Activation: {activate} → Output: {1 if activate else 0}\n")



# Problem 3: Perceptron Comparison (One-vs-All)
print("=== Problem 3: Perceptron Comparison (One-vs-All) ===")

# Input vector
inputs = [0.5, -1, 2, 1, 0]

# Perceptron A
WA = [1.0, -0.5, 0.2, 0.1, 0.0]
BiasA = 0.2
outputA = 0
for i in range(len(inputs)):
    outputA += inputs[i] * WA[i]
print(f"Perceptron A Weighted Sum (before bias): {outputA:.2f}")
outputA += BiasA
print(f"  With bias: {outputA:.2f}")
activateA = outputA >= 0
print(f"  Activation: {activateA} → Output: {1 if activateA else 0}\n")

# Perceptron B
WB = [0.2, 0.2, 0.5, -0.4, 0.3]
BiasB = 0.0
outputB = 0
for i in range(len(inputs)):
    outputB += inputs[i] * WB[i]
print(f"Perceptron B Weighted Sum (before bias): {outputB:.2f}")
outputB += BiasB
print(f"  With bias: {outputB:.2f}")
activateB = outputB >= 0
print(f"  Activation: {activateB} → Output: {1 if activateB else 0}\n")

# Perceptron C
WC = [-0.3, -0.1, 0.4, 0.0, 0.2]
BiasC = -0.6
outputC = 0
for i in range(len(inputs)):
    outputC += inputs[i] * WC[i]
print(f"Perceptron C Weighted Sum (before bias): {outputC:.2f}")
outputC += BiasC
print(f"  With bias: {outputC:.2f}")
activateC = outputC >= 0
print(f"  Activation: {activateC} → Output: {1 if activateC else 0}\n")

# Decide Winner
results = {
    "A": (outputA, activateA),
    "B": (outputB, activateB),
    "C": (outputC, activateC)
}

# Check who fired
active = [k for k,v in results.items() if v[1] == True]

if len(active) == 1:
    print(f"Predicted Class: {active[0]} (Unique Winner)")
elif len(active) > 1:
    winner = max(active, key=lambda k: results[k][0])  # pick highest weighted sum
    print(f"Predicted Class: {winner} (Tie resolved by highest weighted sum)")
else:
    print("No perceptron fired (all outputs = 0)")

