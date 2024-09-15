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
    
def activation_function(a, function_type):  
    if function_type == ActivationFunction.STEP_FUNCTION:
        return 1 if a >= 0 else 0
    elif function_type == ActivationFunction.SIGMOID_FUNCTION:
        return 1 / (1 + math.exp(-a))

input_data = [          #(x1,x2, class)
    ((-0.2, 0.5), 0), 
    ((0.2, -0.7), 0),
    ((0.8, -0.8), 1),
    ((0.8, 1), 1)
]

bias = 0                # w0
weights = (1,1)         #w1, w2

temp_results = apply_weights(input_data, weights, bias)
print(temp_results)