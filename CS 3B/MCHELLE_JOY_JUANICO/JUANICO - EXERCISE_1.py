# 1
print("1. Determine whether the student passes (1) or fails (0)")
print()

weights = [0.6, 0.4]
bias = -3
threshold = 1

test_cases = {
    "(x1, x2) = (8, 7)": [8, 7],
    "(x1, x2) = (3, 4)": [3, 4],
}

for label, inputs in test_cases.items():
    print(f"Inputs: {label}")

    # weighted sum without bias
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]

    # add bias
    output_with_bias = output + bias
    print("Weighted Sum:", output_with_bias)

    # step activation: 1 if net > threshold else 0
    activate = "1 (pass)" if output_with_bias > threshold else "0 (fail)"
    print("Output:", activate)
    print("-----------------------------")

# 2
print()
print("2. Logic Gate Simulation. Given the following setup for a perceptron, compute its output and verify whether it acts as an AND gate.")
print()

inputs = [(0,0), (0,1), (1,0), (1,1)]  
weights = [1, 1]  
bias = -1.5        
threshold = 0

def step_function(x):
    return 1 if x >= threshold else 0 

for x in inputs:
    output = x[0]*weights[0] + x[1]*weights[1] + bias
    
    activated_output = step_function(output)
    
    print(f"Input: {x}")
    print(f"Weighted Sum: {output}")
    print(f"Output: {activated_output}")
    print("-----------------------------")

# 3
print()
print("3. ) Perceptron comparison (One vs All). Given the 3 perceptron, using the same input, compute the output and decided on the predicted class is the WINNER. If a tie is present, compare and get the highest weighted sum.")
print()

input = [0.5, -1, 2, 1, 0]

Wa, ba = [1.0, -0.5, 0.2, 0.1, 0.0], 0.2
Wb, bb = [0.2, 0.2, 0.5, -0.4, 0.3], 0.0
Wc, bc = [-0.3, -0.1, 0.4, 0.0, 0.2], -0.6

def net(w, b, x): 
    s = 0
    for i in range(len(input)):
        s += input[i]*w[i]           
    s += b
    print("Weighted Sum:", s)  
    activate = 1 if s > 0 else 0
    print("Ouput:", activate)
    return s, activate

print("Perceptron A")
na, aa = net(Wa, ba, input)
print("-----------------------------")

print("Perceptron B")
nb, ab = net(Wb, bb,input)
print("-----------------------------")

print("Perceptron C")
nc, ac = net(Wc, bc, input)
print("-----------------------------")

nets = {"A": na, "B": nb, "C": nc}
acts = {"A": aa, "B": ab, "C": ac}

active = [k for k,v in acts.items() if v == 1]
winner = max(active, key=lambda k: nets[k]) if active else max(nets, key=nets.get)
print("Winner:", winner)