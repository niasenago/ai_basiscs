from enum import Enum
import math

class ActivationFunction(Enum):
    STEP_FUNCTION = 1
    SIGMOID_FUNCTION = 2
   
def apply_weights(input_data, weights, bias):
    results = []
    for (x1, x2), _ in input_data:
        result = (x1 * weights[0]) + (x2 * weights[1]) + bias
        results.append(result)
    return results
    
def apply_activation_func(temp_results, function_type):
    results = []
    for elem in temp_results:
        result = activation_function(elem, function_type)
        results.append(result)
    return results

def activation_function(a, function_type):  
    if function_type == ActivationFunction.STEP_FUNCTION:
        return 1 if a >= 0 else 0
    elif function_type == ActivationFunction.SIGMOID_FUNCTION:
        return 1 / (1 + math.exp(-a))

def calculate_accuracy(predictions, ground_truth):
    correct = sum(p == t for p, t in zip(predictions, ground_truth))
    return correct / len(ground_truth)

###############################################
input_data = [          # (x1,x2, class)
    ((-0.2, 0.5), 0), 
    ((0.2, -0.7), 0),
    ((0.8, -0.8), 1),
    ((0.8, 1), 1)
]

# we need to bruteforce these 3 values
bias = - 1.6            # w0
weights = (2,0)         # w1, w2
################################################

if __name__ == "__main__":
    temp_results = apply_weights(input_data, weights, bias)
    print('Choose activation function:\n'
          'For STEP FUNCTION enter 1\n'
          'For SIGMOID FUNCTION enter 2')
    
    activation_function_type_input = int(input())
    activation_function_type = ActivationFunction(activation_function_type_input)

    print(activation_function_type)
    outputs = apply_activation_func(temp_results,activation_function_type )
    print(outputs)