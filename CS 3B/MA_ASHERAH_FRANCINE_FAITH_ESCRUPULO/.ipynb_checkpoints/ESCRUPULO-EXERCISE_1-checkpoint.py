# No. 1 - Student Pass/Fail
inputs1 = [(8, 7), (3, 4)]
weights1 = [0.6, 0.4]
bias1 = -3

threshold1 = 1

print("No. 1 - Student Pass/Fail\n")

for x in inputs1:
    output = 0
    
    for i in range(len(x)):
        output += x[i] * weights1[i]
    
    print("Input:", x)
    print("Output:", round(output, 2))
    
    output_with_bias = output + bias1
    print("Output with bias:", round(output_with_bias, 2))
    
    activate = 1 if output_with_bias >= threshold1 else 0
    print("Activation (Pass=1, Fail=0):", activate, "\n")

# -------------------------

# No. 2 - Logic Gate Simulation (AND Gate)
inputs2 = [(0,0), (0,1), (1,0), (1,1)]
weights2 = [1, 1]
bias2 = -1.5

# threshold = 0
def step_function(x):
    return 1 if x >= 0 else 0

print("No. 2 - Logic Gate Simulation (AND Gate)\n")

for x in inputs2:
    output = 0
    
    for i in range(len(x)):
        output += x[i] * weights2[i]
    
    print("Input:", x)
    print("Output:", output)
    
    output_with_bias = output + bias2
    print("Output with bias:", output_with_bias)
    
    activated = step_function(output_with_bias)
    print("Activation:", activated, "\n")

#3.	(60 points) Perceptron comparison (One vs All).
#  Given the 3 perceptron, using the same input, 
# compute the output and decided on the predicted class is the WINNER.
#  If a tie is present, compare and get the highest weighted sum. 

#################################################################333

inputs = [0.5, -1, 2, 1, 0]
weights_A = [1.0, -0.5, 0.2, 0.1, 0.0]
weights_B = [0.2, 0.2, 0.5, -0.4, 0.3]
weights_C = [-0.3, -0.1, 0.4, 0.0, 0.2]
bias_A = 0.2
bias_B = 0.0
bias_C = -0.6

output_A = 0
output_B = 0
output_C = 0

######################### A ##################
for i in range(len(inputs)):
    output_A += inputs[i] * weights_A[i]

print("Output of A: ", output_A)

output_A += bias_A
print("Output with Bias: ", output_A)

activate = output_A > 0
print("Activation: ", activate)

######################### B ##################
for i in range(len(inputs)):
    output_B += inputs[i] * weights_B[i]

print("Output of B: ", output_B)

output_B += bias_B
print("Output with Bias: ", output_B)

activate = output_B > 0
print("Activation: ", activate)

################## C ######################
for i in range(len(inputs)):
    output_C += inputs[i] * weights_C[i]

print("Output of C: ", output_C)

output_C += bias_C
print("Output with Bias: ", output_C)

activate = output_C > 0
print("Activation: ", activate)


############### Winner #################

if output_A > output_B and output_A > output_C:
    print("Winner is output A")
elif output_B > output_A and output_B > output_C:
    print("Winner is output B")
elif output_C > output_A and output_C > output_B:
    print("Winner is output C")
else:
    
    sum_A = sum([inputs[i] * weights_A[i] for i in range(len(inputs))])
    sum_B = sum([inputs[i] * weights_B[i] for i in range(len(inputs))])
    sum_C = sum([inputs[i] * weights_C[i] for i in range(len(inputs))])

    max_sum = max(sum_A, sum_B, sum_C)
    if max_sum == sum_A:
        print("Winner is output A (by weighted sum)")
    elif max_sum == sum_B:
        print("Winner is output B (by weighted sum)")
    else:
        print("Winner is output C (by weighted sum)")