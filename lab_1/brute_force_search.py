from main import ActivationFunction, apply_weights, apply_activation_func, load_input_data
import math
import random


def calculate_accuracy(predictions, ground_truth):
    correct = sum(p == t for p, t in zip(predictions, ground_truth))
    return correct / len(ground_truth)

def create_range(start, end, step):
    result = []
    current = start
    while current <= end:
        result.append(round(current, 10)) 
        current += step
    if result[-1] != end:
        result.append(end)
    return result

def brute_force_search(input_data, activation_function_type, amount_of_best_combinations):
    top_combinations = []
    num_iter = 0
    biases = create_range(-2.0, 2.0, 0.2)  
    weight_ranges = create_range(-2.0, 2.0, 0.2)
    
    for bias in biases:
        for weight1 in weight_ranges:
            for weight2 in weight_ranges:
                weights = (weight1, weight2)
                temp_results = apply_weights(input_data, weights, bias)
                outputs = apply_activation_func(temp_results, activation_function_type)
                ground_truth = [label for _, label in input_data]
                accuracy = calculate_accuracy(outputs, ground_truth)
                num_iter += 1
                
                if accuracy == 1:
                    top_combinations.append((bias, weights))
                
                if len(top_combinations) == amount_of_best_combinations:
                    break
            if len(top_combinations) == amount_of_best_combinations:
                break
        if len(top_combinations) == amount_of_best_combinations:
            break
    
    if len(top_combinations) == amount_of_best_combinations:
        print("----BRUTE FORCE-------------------")
        print(f"Top {amount_of_best_combinations} best combinations after {num_iter} iterations:")
        for i, (bias, weights) in enumerate(top_combinations, 1):
            print(f"{i}. Bias: {bias:.1f}, w1: {weights[0]:.1f}, w2: {weights[1]:.1f}")
    else:
        print(f"Only {len(top_combinations)} combinations with accuracy 1 found after specified {num_iter} iterations.")
    
def random_brute_force_search(input_data, activation_function_type, amount_of_best_combinations, num_iterations=100_000_00 ):
    top_combinations = []
    num_iter = 0
    biases = create_range(-2.0, 2.0, 0.2)
    weight_ranges = create_range(-2.0, 2.0, 0.2)

    for _ in range(num_iterations):
        bias = random.choice(biases)
        weight1 = random.choice(weight_ranges)
        weight2 = random.choice(weight_ranges)
        weights = (weight1, weight2)

        temp_results = apply_weights(input_data, weights, bias)
        outputs = apply_activation_func(temp_results, activation_function_type)
        ground_truth = [label for _, label in input_data]
        accuracy = calculate_accuracy(outputs, ground_truth)
        num_iter += 1
        
        if accuracy == 1:
            top_combinations.append((bias, weights))

        if len(top_combinations) == amount_of_best_combinations:
            break
    
    if len(top_combinations) == amount_of_best_combinations:
        print("----RANDOM------------------------")
        print(f"Top {amount_of_best_combinations} combinations after {num_iter} iterations:")
        for i, (bias, weights) in enumerate(top_combinations, 1):
            print(f"{i}. Bias: {bias:.3f}, w1: {weights[0]:.3f}, w2: {weights[1]:.3f}")
    else:
        print(f"Only {len(top_combinations)} combinations with accuracy of 1 found after specified {num_iter} iterations.")

    
if __name__ == "__main__":
    input_data = load_input_data('data.json')
    amount_of_combinations = 5
    
    print('Choose activation function:\n'
          'For STEP FUNCTION enter 1\n'
          'For SIGMOID FUNCTION enter 2')
    
    activation_function_type_input = int(input())
    activation_function_type = ActivationFunction(activation_function_type_input)
    brute_force_search(input_data, activation_function_type, amount_of_combinations)
    random_brute_force_search(input_data, activation_function_type, amount_of_combinations)