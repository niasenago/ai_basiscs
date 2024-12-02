import os
import json
import matplotlib.pyplot as plt

def plot_history(history_path, output_dir):
    """
    Generate training and validation performance graphs from a history JSON file.
    Saves the plots as PNG files in the specified output directory.

    Args:
        history_path (str): Path to the model's history JSON file.
        output_dir (str): Directory where the graph will be saved.
    """
    # Load training history
    with open(history_path, 'r') as file:
        history = json.load(file)
    
    # Extract metrics from history
    epochs = range(1, len(history['loss']) + 1)
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    accuracy = history.get('accuracy', [])
    val_accuracy = history.get('val_accuracy', [])
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Mokymo duomenų paklaida')
    if val_loss:
        plt.plot(epochs, val_loss, label='Validavimo duomenų paklaida')
    plt.title('Neuroninio tinklo paklaida')
    plt.xlabel('Epochos')
    plt.ylabel('Paklaida')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # Plot Accuracy
    if accuracy:  # Only plot if accuracy is available
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracy, label='Mokymo duomenų tikslumas')
        if val_accuracy:
            plt.plot(epochs, val_accuracy, label='Validavimo duomenų tikslumas')
        plt.title('Neuroninio tinklo tikslumas')
        plt.xlabel('Epochos')
        plt.ylabel('Tikslumas')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
        plt.close()

def visualize_results(results_dir):
    """
    Iterates through the results folder and generates plots for each model.

    Args:
        results_dir (str): Path to the directory containing model results.
    """
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            history_file = os.path.join(model_path, f"{model_dir}_history.json")
            if os.path.exists(history_file):
                print(f"Processing: {history_file}")
                plot_history(history_file, model_path)
            else:
                print(f"History file not found: {history_file}")

if __name__ == "__main__":
    # Directory containing model result folders
    results_directory = "results"
    visualize_results(results_directory)
