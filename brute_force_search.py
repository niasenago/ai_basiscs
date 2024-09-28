from main import ActivationFunction, apply_weights, apply_activation_func, load_input_data
import math
import random


def calculate_accuracy(predictions, ground_truth):
    correct = sum(p == t for p, t in zip(predictions, ground_truth))
    return correct / len(ground_truth)

def brute_force_search(input_data, activation_function_type):
    best_accuracy = 0
    best_bias = 0
    best_weights = (0, 0)
    num_iter = 0
    
    biases = [i * 0.2 for i in range(-10, 11)]  # Example range of bias values
    weight_ranges = [i * 0.2 for i in range(-10, 11)]  # Example range of weight values
    for bias in biases:
        for weight1 in weight_ranges:
            for weight2 in weight_ranges:
                weights = (weight1, weight2)
                temp_results = apply_weights(input_data, weights, bias)
                outputs = apply_activation_func(temp_results, activation_function_type)
                ground_truth = [label for _, label in input_data]
                accuracy = calculate_accuracy(outputs, ground_truth)
                num_iter += 1
                if accuracy > best_accuracy:
                    print(f"Best accuracy so far: {accuracy}")
                    print(f"w0: {bias}; w1 {weights[0]}; w2 {weights[1]}")

                    best_accuracy = accuracy
                    best_bias = bias
                    best_weights = weights
    print(f"Num of iterations: {num_iter}")
    return best_bias, best_weights, best_accuracy


def random_brute_force_search(input_data, activation_function_type, num_iterations=100_000_00):
    best_accuracy = 0
    best_bias = 0
    best_weights = (0, 0)
    num_iter = 0
    # Loop for a number of random iterations
    # Loop for a number of random iterations
    for _ in range(num_iterations):
        # Randomly select bias and weights as integers in the range from -10 to 11, and scale by 0.2
        bias = random.randint(-10, 11) * 0.2
        weight1 = random.randint(-10, 11) * 0.2
        weight2 = random.randint(-10, 11) * 0.2
        weights = (weight1, weight2)

        # Apply weights and bias to input data
        temp_results = apply_weights(input_data, weights, bias)
        outputs = apply_activation_func(temp_results, activation_function_type)
        ground_truth = [label for _, label in input_data]
        accuracy = calculate_accuracy(outputs, ground_truth)
        num_iter += 1
        # Update if we find better accuracy
        if accuracy > best_accuracy:
            print(f"Best accuracy so far: {accuracy}")
            print(f"w0: {bias}; w1 {weights[0]}; w2 {weights[1]}")
            best_accuracy = accuracy
            best_bias = bias
            best_weights = weights
            if accuracy == 1:
                break
    print(f"Rand Num of iterations: {num_iter}")

    return best_bias, best_weights, best_accuracy
    
if __name__ == "__main__":
    input_data = load_input_data('data.json')

    activation_function_type = ActivationFunction.STEP_FUNCTION
    best_bias, best_weights, best_accuracy = brute_force_search(input_data, activation_function_type)
    
    print(f'Best Bias: {best_bias}')
    print(f'Best Weights: {best_weights}')
    print(f'Best Accuracy: {best_accuracy}')
    best_bias, best_weights, best_accuracy = random_brute_force_search(input_data, activation_function_type)
    
    print(f'Rand Best Bias: {best_bias}')
    print(f'Rand Best Weights: {best_weights}')
    print(f'Rand Best Accuracy: {best_accuracy}')