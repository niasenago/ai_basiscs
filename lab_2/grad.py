import json
import pandas as pd
import numpy as np

from visualize_data import DatasetChoice

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# source: https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
def sigmoid_derivative(a):
    return sigmoid(a) * (1 - sigmoid(a))


def read_parameters(config_file, dataset_choice):
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found.")
        return None
    except json.JSONDecodeError:
        print("Error reading the config file.")
        return None
    
    # get hyperparameters
    hyperparams = config.get("hyperparameters", {})
    learning_rate = hyperparams.get("learning_rate")  
    epoch_count = hyperparams.get("epoch_count") 
    wanted_accuracy = hyperparams.get("wanted_accuracy")  
    
    # get data
    if dataset_choice == DatasetChoice.IRIS:
        dataset_config = config["datasets"]["IRIS"]
    elif dataset_choice == DatasetChoice.BREAST_CANCER:
        dataset_config = config["datasets"]["BREAST_CANCER"]

    url = dataset_config["url_for_learning"]

    return learning_rate, epoch_count, wanted_accuracy, url

def evaluate_performance(predictions, target_class, epoch):
    total_error = 0

    # MSE
    for i in range(len(predictions)):
        error = (predictions[i] - target_class.iloc[i]) ** 2
        total_error += error
    total_error = total_error / len(predictions) 

    # the proportion of correct predictions aka accuracy 
    accuracy = np.mean((predictions >= 0.5) == target_class)

    return {
        'epoch': epoch + 1,
        'totalError': total_error,
        'accuracy': accuracy
    }



###
# Returns best weights, biases, thier accuracy and dataframe with epoch, accuracy and totalError 
##
def run_grad_descent(input_df, epoch_count, wanted_accuracy, learning_rate):
    input_data = input_df.iloc[:, :-1]  # Attributes (all columns except last)
    target_class = input_df.iloc[:, -1]  # Target class (last column)

    # Initialize weights and bias
    weights = 2 * np.random.rand(input_data.shape[1]) - 1 # Number of weights equals number of attributes
    bias = 2 * np.random.rand() - 1

    performance_data = []

    for epoch in range(epoch_count):
        weights, bias = descend(input_data, weights, bias, learning_rate, target_class)

        predictions = sigmoid(np.dot(input_data, weights) + bias)

        performance = evaluate_performance(predictions, target_class, epoch)
        performance_data.append(performance)

        if epoch % 20 == 0:
            print(f"Epoch {performance['epoch']}, Total Error: {performance['totalError']}, Accuracy: {performance['accuracy']}")

        # Stop early if desired accuracy is reached
        if performance['accuracy'] >= wanted_accuracy:
            print(f"Desired accuracy reached after {performance['epoch']} epochs")
            break
    performance_df = pd.DataFrame(performance_data)        
    
    return weights, bias, performance['accuracy'], performance_df

# Neurono mokymui naudoti stochastinį gradientinį nusileidimą ir sigmoidinį neuroną.
def descend(input_data, weights, bias, learning_rate, target_class):
    for i in range(len(input_data)):
        inputs = input_data.iloc[i].values  # Extract row as array
        target = target_class.iloc[i]
        
        # compute weighted sum and activation
        weighted_sum = np.dot(inputs, weights) + bias
        prediction = sigmoid(weighted_sum)
        
        # Compute error: this is part of SGD
        error = prediction - target
        
        # Backpropagation: compute gradients
        dZ = error * sigmoid_derivative(weighted_sum)
        # dW = error * sigmoid' * [x1, x2 .. xn]; in other words it's a gradient of weights
        dW = np.dot(np.array(inputs).T, dZ) 
        dB = dZ  # Gradient of bias
        
        # Update weights and bias
        weights -= learning_rate * dW
        bias -= learning_rate * dB
        
    return weights, bias


def read_data_into_dataframe(filename):
    df = pd.read_csv(filename, header=None)    
    return df

def save_results(dataset_choice_input, weights, bias, performance_df):
    results_dir = "./results"

    weights_file = f"{results_dir}/{dataset_choice_input}.weights.json"
    performance_file = f"{results_dir}/{dataset_choice_input}.learning.performance.csv"

    weights_data = {
        'weights': weights.tolist(),  # Convert NumPy array to list
        'bias': bias
    }
    with open(weights_file, 'w') as f:
        json.dump(weights_data, f, indent=4)

    print(f"Weights and bias saved to {weights_file}")

    performance_df.to_csv(performance_file, index=False)
    print(f"Performance data saved to {performance_file}")


def main ():
    config_file = "config.json"

    try:
        dataset_choice_input = input("Choose dataset ('iris' or 'breast-cancer'): ").strip().lower()
        dataset_choice = DatasetChoice(dataset_choice_input)
    except ValueError:
        print("Invalid dataset choice. Please choose 'iris' or 'breast-cancer'.")
        return

    learning_rate, epoch_count, wanted_accuracy, url = read_parameters(config_file, dataset_choice)

    print(f"Learning Rate: {learning_rate}")
    print(f"Epoch Count: {epoch_count}")
    print(f"Wanted Accuracy: {wanted_accuracy}")

    df = read_data_into_dataframe(url)

    print("Initial Data:")
    print(df.head())  # Show the first few rows for confirmation

    # Run gradient descent on the dataset
    weights, bias, accuracy, performance_df = run_grad_descent(df, epoch_count, wanted_accuracy, learning_rate)

    print(f"Final Weights: {weights}")
    print(f"Final Bias: {bias}")
    print(f"Final Accuracy: {accuracy}")
    print(f"Peroformace:\n{performance_df}")
    save_results(dataset_choice_input, weights, bias, performance_df)    


if __name__ == "__main__":
    main()