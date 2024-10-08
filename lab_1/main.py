from enum import Enum
import math
import json

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
        sigmoid_value = 1 / (1 + math.exp(-a))
        return 1 if sigmoid_value >= 0.5 else 0

def load_input_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        input_data = [((item["inputs"][0], item["inputs"][1]), item["class"]) for item in data["input_data"]]
        return input_data

def load_weights(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        bias = data["bias"]
        weights = tuple(data["weights"])
        return weights, bias
    

if __name__ == "__main__":
    # (x1,x2, class)
    input_data = load_input_data('data.json')
   
    # w1, w2, w0
    weights, bias = load_weights('weights.json')    
    temp_results = apply_weights(input_data, weights, bias)
    print('Choose activation function:\n'
          'For STEP FUNCTION enter 1\n'
          'For SIGMOID FUNCTION enter 2')
    
    activation_function_type_input = int(input())
    activation_function_type = ActivationFunction(activation_function_type_input)

    print(activation_function_type)
    outputs = apply_activation_func(temp_results,activation_function_type )
    print(outputs)