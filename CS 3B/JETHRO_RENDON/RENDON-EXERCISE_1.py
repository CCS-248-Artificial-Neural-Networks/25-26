# 1. (10 points) Determine whether the student passes (1) or fails (0)based on:
# a. Hours studied
# b. Hours of sleep

def evaluate_students(name, hours_studied, hours_slept, bias):
    inputs = [hours_studied, hours_slept]
    weights = [0.6, 0.4]
    output = 0
    
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]   
    
    output += bias 
        
    activate = output > 1
    activation_value = 0
    
    if activate:
        activation_value = 1
    else:
        activation_value = 0
    
    print(f"Name: {name}")
    print(f"Inputs: {inputs[0]}, {inputs[1]}")
    print(f"Output: {output}")
    print(f"Activation -> {activation_value}")
    
    if activate:
        print("Evaluation: Passed!\n")
    else:
        print("Evaluation: Failed :((\n")

# 2. (20 points) Logic Gate Simulation. Given the following setup for a perceptron, compute its output and verify whether it acts as an AND gate.

def logic_gate_simulation(input_num, x1, x2, bias):
    inputs = [x1, x2]
    weights = [1, 1]
    output = 0
    
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
        
    output += bias
    activate = output > 0
    
    activation_value = 0
    
    if activate:
        activation_value = 1
    else:
        activation_value = 0
        
    if x1 == 1 and x2 == 1:
        is_and_gate = True
    else:
        is_and_gate = False
    
    print(f"\n{input_num}. Inputs: {x1}, {x2}")
    print(f"Output: {output}")
    print(f"Activation -> {activation_value}")
    print(f"Acts as an AND gate: {is_and_gate}")
    
    
# 3. (60 points) Perceptron comparison (One vs All). Given the 3 perceptron, using the same input, compute the output and decided on the predicted class is the WINNER. If a tie is present, compare and get the highest weighted sum.

# Given weights
weights_a = [1.0, -0.5, 0.2, 0.1, 0.0]
weights_b = [0.2, 0.2, 0.5, -0.4, 0.3]
weights_c = [-0.3, -0.1, 0.4, 0.0, 0.2]

def perceptron_comparison(perceptron, weights, bias):
    inputs = [0.5, -1, 2, 1, 0]
    output = 0
    
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]

    output += bias
    print(f"Output of Perceptron {perceptron}: {output}")
    return output
    
print("= = = EXERCISE FOR UNIT 1 = = =")

cont = True

while cont:
    print("\nnote: Choose an option to see evaluation." + "\n" +
        "1. Student Evaluation" + "\n" +
        "2. Logic Gate Simulation" + "\n" +
        "3. Perceptron Comparison" + "\n" +
        "4. Exit" + "\n")


    # main
    choice = input("Choose an option (1-4): ").strip()

    if choice == '1':
        print("\n= = = Student Evaluation = = =")
        evaluate_students("Jethro", 8, 7, -3)
        evaluate_students("Joeross", 3, 4, -3)

    elif choice == '2':
        print("\n= = = Logic Gates Simulation = = =")
        logic_gate_simulation(1, 0, 0, -1.5)
        logic_gate_simulation(2, 0, 1, -1.5)
        logic_gate_simulation(3, 1, 0, -1.5)
        logic_gate_simulation(4, 1, 1, -1.5)
        print("= = = = = = = = = = = = = = = = = =")

    elif choice == '3':
        print("\n= = = Perceptron Comparison = = =")
        weights_a = [1.0, -0.5, 0.2, 0.1, 0.0]
        weights_b = [0.2, 0.2, 0.5, -0.4, 0.3]
        weights_c = [-0.3, -0.1, 0.4, 0.0, 0.2]

        w_a = perceptron_comparison('A', weights_a, 0.2)
        w_b = perceptron_comparison('B', weights_b, 0.0)
        w_c = perceptron_comparison('C', weights_c, -0.6)

        max_w = w_a
        winner = "A"

        if w_b > max_w:
            max_w = w_b
            winner = "B"
        if w_c > max_w:
            max_w = w_c
            winner = "C"

        print(f"\nThe Perceptron Winner is perceptron {winner} with an output of {max_w}")
    
    elif choice == '4':
        print("\n\nJethro A. Rendon & Joeross Palabrica Assignment")
        cont = False
        
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")