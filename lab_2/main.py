import json
import pandas as pd
import numpy as np
from grad import sigmoid, evaluate_performance  

def load_weights(dataset_choice_input):
    weights_file = f"./results/{dataset_choice_input}.weights.json"
    try:
        with open(weights_file, 'r') as f:
            weights_data = json.load(f)
            weights = np.array(weights_data['weights'])  # Convert list back to NumPy array
            bias = weights_data['bias']
        return weights, bias
    except FileNotFoundError:
        print(f"Error: Weights file {weights_file} not found.")
        return None, None

def load_test_data(dataset_choice_input):
    if dataset_choice_input == 'iris':
        test_data_file = "./data/iris.augmented.4tests.data"
    elif dataset_choice_input == 'breast-cancer':
        test_data_file = "./data/breast-cancer-wisconsin-converted.4tests.data"
    else:
        print("Invalid dataset choice.")
        return None

    try:
        df = pd.read_csv(test_data_file, header=None)
        return df
    except FileNotFoundError:
        print(f"Error: Test data file {test_data_file} not found.")
        return None

def apply_weights(df, weights, bias):
    input_data = df.iloc[:, :-1]  # Attributes 
    target_class = df.iloc[:, -1]  # Target class 
    
    # Apply the sigmoid function to weighted sums
    predictions = sigmoid(np.dot(input_data, weights) + bias)
    
    return predictions, target_class

def main():
    dataset_choice_input = input("Choose dataset ('iris' or 'breast-cancer'): ").strip().lower()
    
    weights, bias = load_weights(dataset_choice_input)
    if weights is None or bias is None:
        return
    
    df = load_test_data(dataset_choice_input)
    if df is None:
        return
    
    # Apply weights to the test data
    predictions, target_class = apply_weights(df, weights, bias)
    
    # Evaluate performance
    performance = evaluate_performance(predictions, target_class, epoch=0)  # epoch=0 for test
    print(f"Test Performance - Total Error: {performance['totalError']}, Accuracy: {performance['accuracy']}")

if __name__ == "__main__":
    main()
