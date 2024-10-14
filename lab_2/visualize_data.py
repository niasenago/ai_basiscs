import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from enum import Enum
import json

# inspired by:https://builtin.com/machine-learning/pca-in-python

class DatasetChoice(Enum):
    IRIS = 'iris'
    BREAST_CANCER = 'breast-cancer'

# Function to perform dimensionality reduction using PCA
def flatten_data(data):
    # PCA algorithm performs better when values are onto unit scale (mean = 0 and variance = 1)
    # Mean is the "center" of the data
    # Standard deviation measures the "spread" or dispersion of the data around the mean.     
    x = StandardScaler().fit_transform(data)

    # Perform PCA to reduce dimensionality to 2
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data=principalComponents, columns=['Feature 1', 'Feature 2'])
    return principalDf


def visualize_data(principalDf, target):
    finalDf = pd.concat([principalDf, target], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Feature 1', fontsize=15)
    ax.set_ylabel('Feature 2', fontsize=15)
    ax.set_title('2 dimensional dataset', fontsize=20)

    targets = [0, 1]
    colors = ['r', 'g']

    # Create the plot itself
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Feature 1'],
                   finalDf.loc[indicesToKeep, 'Feature 2'],
                   c=color,
                   s=25)

    ax.legend(targets)
    ax.grid()

    plt.show()

def main():
    # Load the config file
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found. Please ensure the config.json file exists.")
        return
    except json.JSONDecodeError:
        print("Error reading the config file. Ensure it is valid JSON.")
        return

    try:
        dataset_choice_input = input("Choose dataset ('iris' or 'breast-cancer'): ").strip().lower()
        dataset_choice = DatasetChoice(dataset_choice_input)
    except ValueError:
        print("Invalid dataset choice. Please choose 'iris' or 'breast-cancer'.")
        return

    # Load the dataset URL and feature names from config
    if dataset_choice == DatasetChoice.IRIS:
        dataset_config = config["datasets"]["IRIS"]
    elif dataset_choice == DatasetChoice.BREAST_CANCER:
        dataset_config = config["datasets"]["BREAST_CANCER"]

    url = dataset_config["url"]
    feature_names = dataset_config["feature_names"]

    try:
        df = pd.read_csv(url, names=(feature_names + ['target']))
    except FileNotFoundError:
        print(f"Dataset file not found at {url}.")
        return
    except pd.errors.ParserError:
        print(f"Error parsing the dataset file at {url}.")
        return

    # Separate out features and target
    features = df[feature_names]
    target = df[['target']]

    # Perform dimension resuction and visualize the results
    principalDf = flatten_data(features)
    visualize_data(principalDf, target)

if __name__ == "__main__":
    main()
