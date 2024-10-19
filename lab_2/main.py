import numpy as np
from enum import Enum
import math


class ActivationFunction(Enum):
    STEP_FUNCTION = 1
    SIGMOID_FUNCTION = 2 # we will this function in my case

def apply_weights(input_data, weights, bias): # input data is a dataframe with atribute values and target
    inputs = np.array([i for i, _ in input_data]) 
    weights = np.array(weights)
    bias = np.array(bias)

    # Perform matrix multiplication and add the bias
    results = np.dot(inputs, weights) + bias
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
        sigmoid_value = 1 / (1 + math.exp(-a))
        return 1 if sigmoid_value >= 0.5 else 0

if __name__ == "__main__":
    print("hello world")
    # initial values for wieghts and bias are generated randomly in range [-1;1]

    # Tada gradientinio nusileidimo algoritmu judama
    # antigradiento kryptimi, svorių reikšmes
    # keičiant pagal iteracinę formulę