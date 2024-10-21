import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from enum import Enum
import json

from visualize_data import DatasetChoice
def load_performance_data(dataset_choice):
    if dataset_choice == DatasetChoice.IRIS:
        performance_file = "./results/iris.learning.performance.csv"
    elif dataset_choice == DatasetChoice.BREAST_CANCER:
        performance_file = "./results/breast-cancer.learning.performance.csv"
    else:
        print("Invalid dataset choice for performance data.")
        return None

    try:
        # Load the CSV data into a pandas DataFrame
        performance_df = pd.read_csv(performance_file)
        return performance_df
    except FileNotFoundError:
        print(f"Performance data file not found: {performance_file}")
        return None

def plot_performance_data(performance_df, dataset_choice):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Total Error vs. Epoch
    axs[0].plot(performance_df['epoch'], performance_df['totalError'], color='blue', label='Paklaida')
    axs[0].set_title(f"Paklaida ({dataset_choice.value})")
    axs[0].set_xlabel('Epocha')
    axs[0].set_ylabel('Paklaida')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy vs. Epoch
    axs[1].plot(performance_df['epoch'], performance_df['accuracy'], color='green', label='Klasifikavimo tikslumas')
    axs[1].set_title(f"Klasifikavimo tikslumas ({dataset_choice.value})")
    axs[1].set_xlabel('Epocha')
    axs[1].set_ylabel('Klasifikavimo tikslumas')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
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

    # Load performance data for the chosen dataset
    performance_df = load_performance_data(dataset_choice)
    if performance_df is None:
        return

    # Plot the performance data
    plot_performance_data(performance_df, dataset_choice)
if __name__ == "__main__":
    main()
