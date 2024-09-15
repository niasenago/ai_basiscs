from main import ActivationFunction, apply_weights, apply_activation_func, calculate_accuracy
import itertools
import math

def brute_force_search(input_data, activation_function_type):
    best_accuracy = 0
    best_bias = 0
    best_weights = (0, 0)
    
    biases = [i * 0.2 for i in range(-10, 11)]  # Example range of bias values
    weight_ranges = [i * 0.2 for i in range(-10, 11)]  # Example range of weight values

    for bias in biases:
        for weight1, weight2 in itertools.product(weight_ranges, repeat=2):
            weights = (weight1, weight2)
            temp_results = apply_weights(input_data, weights, bias)
            outputs = apply_activation_func(temp_results, activation_function_type)
            ground_truth = [label for _, label in input_data]
            accuracy = calculate_accuracy(outputs, ground_truth)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_bias = bias
                best_weights = weights

    return best_bias, best_weights, best_accuracy

if __name__ == "__main__":
    input_data = [          # (x1,x2, class)
        ((-0.2, 0.5), 0), 
        ((0.2, -0.7), 0),
        ((0.8, -0.8), 1),
        ((0.8, 1), 1)
    ]

    activation_function_type = ActivationFunction.STEP_FUNCTION
    best_bias, best_weights, best_accuracy = brute_force_search(input_data, activation_function_type)
    
    print(f'Best Bias: {best_bias}')
    print(f'Best Weights: {best_weights}')
    print(f'Best Accuracy: {best_accuracy}')
