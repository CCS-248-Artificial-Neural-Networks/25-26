# Perception class for simulating a simple neural network
class Perception:
    def __init__(self, inputs, weights, bias, step_threshold):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.step_threshold = step_threshold
    
    def compute_weighted_sum(self):
        weighted_sum = sum(self.inputs[i] * self.weights[i] for i in range(len(self.inputs)))
        return weighted_sum + self.bias

    def activate(self, weighted_sum):
        if weighted_sum > self.step_threshold:
            return 1
        return 0



"""
1. 
Determine whether the student passes (1) or fails (0) based on:
•	Hours studied
•	Hours of sleep
Given weights w1 = 0.6, w2 = 0.4, bias = -3
Step function with a threshold of 1 
Predict the output of the following inputs:
1.	(x1, x2) = (8, 7)
2.	(x1, x2) = (3, 4)
"""

perception_1_1 = Perception([8, 7], [0.6, 0.4], -3, 1)
perception_1_2 = Perception([3, 4], [0.6, 0.4], -3, 1)

print("--== PROBLEM 1 ==--")
weighted_sum_1_1 = perception_1_1.compute_weighted_sum()
weighted_sum_1_2 = perception_1_2.compute_weighted_sum()
activation_1_1 = perception_1_1.activate(weighted_sum_1_1)
activation_1_2 = perception_1_2.activate(weighted_sum_1_2)
print(f"Output for \"Hours studied\" (8, 7): {activation_1_1} with weighted sum {weighted_sum_1_1} \n - Student {'passes' if activation_1_1 == 1 else 'fails'}")
print(f"Output for \"Hours of sleep\" (3, 4): {activation_1_2} with weighted sum {weighted_sum_1_2} \n - Student {'passes' if activation_1_2 == 1 else 'fails'}")
print()



"""
2. 
Logic Gate Simulation. Given the following setup for a perceptron, 
compute its output and verify whether it acts as an AND gate.
Given weights w1 = 1, w2 = 1, bias = -1.5
Step function with a threshold of 0
Inputs:
	(0,0), (0,1), (1,0), (1,1)
"""

perception_2_1 = Perception([0, 0], [1, 1], -1.5, 0) # (0, 0)
perception_2_2 = Perception([0, 1], [1, 1], -1.5, 0) # (0, 1)
perception_2_3 = Perception([1, 0], [1, 1], -1.5, 0) # (0, 2)
perception_2_4 = Perception([1, 1], [1, 1], -1.5, 0) # (0, 3)

print("--== PROBLEM 2 ==--")
weighted_sums = [
    perception_2_1.compute_weighted_sum(),
    perception_2_2.compute_weighted_sum(),
    perception_2_3.compute_weighted_sum(),
    perception_2_4.compute_weighted_sum()
]
outputs = [
    perception_2_1.activate(weighted_sums[0]),
    perception_2_2.activate(weighted_sums[1]),
    perception_2_3.activate(weighted_sums[2]),
    perception_2_4.activate(weighted_sums[3])
]
inputs = [(0,0), (0,1), (1,0), (1,1)]
for inp, out, w_sum in zip(inputs, outputs, weighted_sums):
    print(f"Output for {inp}: {out} with weighted sum {w_sum}")

# AND gate truth table: [0, 0, 0, 1]
if outputs == [0, 0, 0, 1]:
    print("This perceptron acts as an AND gate.")
else:
    print("This perceptron does NOT act as an AND gate.")
print()



"""
3.
Perceptron comparison (One vs All). Given the 3 perceptron, using the same input, 
compute the output and decided on the predicted class is the WINNER. 
If a tie is present, compare and get the highest weighted sum. 
Step function with a threshold of 0

Inputs = [0.5, -1, 2, 1, 0]
Perceptron A Configuration:
•	Weights: WA = [1.0, -0.5, 0.2, 0.1, 0.0]
•	BiasA: 0.2
Perceptron B
•	Weights: WB = [0.2, 0.2, 0.5, -0.4, 0.3]
•	BiasB: 0.0
Perceptron C
•	Weights: WC = [-0.3, -0.1, 0.4, 0.0, 0.2]
•	BiasC: -0.6
"""

perception_3_1 = Perception([0.5, -1, 2, 1, 0], [1.0, -0.5, 0.2, 0.1, 0.0], 0.2, 0)
perception_3_2 = Perception([0.5, -1, 2, 1, 0], [0.2, 0.2, 0.5, -0.4, 0.3], 0.0, 0)
perception_3_3 = Perception([0.5, -1, 2, 1, 0], [-0.3, -0.1, 0.4, 0.0, 0.2], -0.6, 0)

weighted_sum_3_1 = perception_3_1.compute_weighted_sum()
weighted_sum_3_2 = perception_3_2.compute_weighted_sum()
weighted_sum_3_3 = perception_3_3.compute_weighted_sum()
activation_3_1 = perception_3_1.activate(weighted_sum_3_1)
activation_3_2 = perception_3_2.activate(weighted_sum_3_2)
activation_3_3 = perception_3_3.activate(weighted_sum_3_3)

print("--== PROBLEM 3 ==--")
print(f"Output for Perceptron A: {activation_3_1} with weighted sum {weighted_sum_3_1}")
print(f"Output for Perceptron B: {activation_3_2} with weighted sum {weighted_sum_3_2}")
print(f"Output for Perceptron C: {activation_3_3} with weighted sum {weighted_sum_3_3}")

# Determine the winner
activations = [activation_3_1, activation_3_2, activation_3_3]
weighted_sums = [weighted_sum_3_1, weighted_sum_3_2, weighted_sum_3_3]
perceptron_labels = ['A', 'B', 'C']

max_activation = max(activations)
indices = [i for i, act in enumerate(activations) if act == max_activation]

if len(indices) == 1:
    winner_label = perceptron_labels[indices[0]]
else:
    max_sum = float('-inf')
    winner_index = None
    for i in indices:
        if weighted_sums[i] > max_sum:
            max_sum = weighted_sums[i]
            winner_index = i
    winner_label = perceptron_labels[winner_index]

print(f"Winner is: Perceptron {winner_label} with weighted sum {max_sum}")
print()
