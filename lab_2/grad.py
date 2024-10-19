import json
import pandas as pd
import numpy as np

from visualize_data import DatasetChoice

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

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

    url = dataset_config["url"]

    return learning_rate, epoch_count, wanted_accuracy, url

# Mokymas stabdomas arba atlikus iš anksto nustatytą iteracijų (ar epochų) skaičių, arba pasiekus norimą mažą paklaidos reikšmę.
def run_grad_descent(input_df, epoch_count, wanted_accuracy, learning_rate):
    input_data = input_df.iloc[:, :-1]  # Attributes (all columns except last)
    target_class = input_df.iloc[:, -1]  # Target class (last column)

    # Initialize weights and bias
    weights = np.random.rand(input_data.shape[1])  # Number of weights equals number of attributes
    bias = np.random.rand()

    for epoch in range(epoch_count):
        weights, bias = descend(input_data, weights, bias, learning_rate, target_class)

        # Compute accuracy after each epoch
        predictions = sigmoid(np.dot(input_data, weights) + bias)
        accuracy = np.mean((predictions >= 0.5) == target_class)
        
        # Check if desired accuracy is reached
        if accuracy >= wanted_accuracy:
            break
    
    return weights, bias, accuracy

# Neurono mokymui naudoti stochastinį gradientinį nusileidimą ir sigmoidinį neuroną.
def descend(input_data, weights, bias, learning_rate, target_class):
    for i in range(len(input_data)):
        inputs = input_data.iloc[i].values  # Extract row as array
        target = target_class.iloc[i]
        
        # compute weighted sum and activation
        z = np.dot(inputs, weights) + bias
        prediction = sigmoid(z) # ???
        
        # Compute error
        error = prediction - target
        
        # Backpropagation: compute gradients
        dZ = error * sigmoid_derivative(z)
        dW = np.dot(np.array(inputs).T, dZ)  # Gradient of weights
        dB = dZ  # Gradient of bias
        
        # Update weights and bias
        weights -= learning_rate * dW
        bias -= learning_rate * dB
        
    return weights, bias




def read_data_into_dataframe(filename):
    df = pd.read_csv(filename, header=None)    
    return df

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
    weights, bias, accuracy = run_grad_descent(df, epoch_count, wanted_accuracy, learning_rate)

    print(f"Final Weights: {weights}")
    print(f"Final Bias: {bias}")
    print(f"Final Accuracy: {accuracy}")


if __name__ == "__main__":
    main()