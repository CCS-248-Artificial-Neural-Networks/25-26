#1

inputs = [8, 7]
weights = [0.6, 0.4]
bias = -3

output = 0
for i in range(len(inputs)):
    output += inputs[i] * weights[i]

print ("Ouput: ", output)

output += bias
print("Output with bias: " , output)

activate = output > 1
print ("Activation: ", activate)


# inputs = [3, 4]
# weights = [0.6, 0.4]
# bias = -3

# output = 0
# for i in range(len(inputs)):
#     output += inputs[i] * weights[i]

# print ("Ouput: ", output)

# output += bias
# print("Output with bias: " , output)

# activate = output > 1
# print ("Activation: ", activate)

# 2

# inputs = [0, 0]
# weights = [1, 1]
# bias = -1.5

# output = 0
# for i in range(len(inputs)):
#     output += inputs[i] * weights[i]

# print ("Output: ", output)

# output += bias
# print("Output with bias: ", output)

# activate = output > 0
# print ("Activation: ", activate)


# inputs = [0, 1]
# weights = [1, 1]
# bias = -1.5

# output = 0
# for i in range(len(inputs)):
#     output += inputs [i] * weights[i]

# print("Output: ", output)


# output += bias
# print("Output with bias: ", output)

# activate = output > 0
# print("Activation: ", activate)


# inputs = [1, 0]
# weights = [1, 1]
# bias = -1.5

# output = 0
# for i in range(len(inputs)):
#     output += inputs [i] * weights[i]

# print("Output: ", output)


# output += bias
# print("Output with bias: ", output)

# activate = output > 0
# print("Activation: ", activate)


# inputs = [1, 1]
# weights = [1, 1]
# bias = -1.5

# output = 0
# for i in range(len(inputs)):
#     output += inputs [i] * weights[i]

# print("Output: ", output)


# output += bias
# print("Output with bias: ", output)

# activate = output > 0
# print("Activation: ", activate)


#3

# inputs = [0.5, -1, 2, 1, 0]
# weights = [1.0, -0.5, 0.2, 0.1, 0.0]
# bias = 0.2

# output = 0
# for i in range(len(inputs)):
#     output += inputs [i] * weights[i]

# print("Output for A: ", output)


# output += bias
# print("Output with bias for A: ", output)

# activateA = output > 0
# print("Activation for A: ", activateA)


# inputs = [0.5, -1, 2, 1, 0]
# weights = [0.2, 0.2, 0.5, -0.4, 0.3]
# bias = 0.0

# output = 0
# for i in range(len(inputs)):
#     output += inputs [i] * weights[i]

# print("Output for B: ", output)


# output += bias
# print("Output with bias for B: ", output)

# activateB = output > 0
# print("Activation for B: ", activateB)


# inputs = [0.5, -1, 2, 1, 0]
# weights = [-0.3, -0.1, 0.4, 0.0, 0.2]
# bias = -0.6

# output = 0
# for i in range(len(inputs)):
#     output += inputs [i] * weights[i]

# print("Output for  C: ", output)


# output += bias
# print("Output with bias for C: ", output)

# activateC = output > 0
# print("Activation for C: ", activateC)


# outputs = {
#     "A": (activateA, 0.5*1.0 + (-1)*(-0.5) + 2*0.2 + 1*0.1 + 0*0.0 + 0.2),
#     "B": (activateB, 0.5*0.2 + (-1)*0.2 + 2*0.5 + 1*(-0.4) + 0*0.3 + 0.0),
#     "C": (activateC, 0.5*(-0.3) + (-1)*(-0.1) + 2*0.4 + 1*0.0 + 0*0.2 - 0.6)
# }

# active_outputs = {k: v[1] for k, v in outputs.items() if v[0]}

# if active_outputs:
#     winner = max(active_outputs, key=active_outputs.get)
# else:
#     winner = "No Winner (all inactive)"

# print("Predicted Class (Winner): ", winner)