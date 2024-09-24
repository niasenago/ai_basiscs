import json
import matplotlib.pyplot as plt
from main import load_input_data, load_weights

def visualize_input_data(input_data, best_weights, best_bias):
    # Separate inputs and their corresponding classes
    class_0_data = [inputs for inputs, label in input_data if label == 0]
    class_1_data = [inputs for inputs, label in input_data if label == 1]

    # Unpack the inputs for plotting
    class_0_x = [x1 for x1, _ in class_0_data]
    class_0_y = [x2 for _, x2 in class_0_data]

    class_1_x = [x1 for x1, _ in class_1_data]
    class_1_y = [x2 for _, x2 in class_1_data]

    # Plot the data
    plt.scatter(class_0_x, class_0_y, color='blue', label='Class 0', marker='o')
    plt.scatter(class_1_x, class_1_y, color='red', label='Class 1', marker='x')

    # Plot the decision boundary line based on best_weights and best_bias
    w1, w2 = best_weights
    bias = best_bias

    if w2 != 0:  # Avoid division by zero if w2 is zero
        # Solve for x2 as a function of x1: x2 = -(w1 * x1 + bias) / w2
        x_values = [-1, 1]  # Example x1 range for plotting the boundary
        y_values = [-(w1 * x + bias) / w2 for x in x_values]
        plt.plot(x_values, y_values, color='green', linestyle='--', label='Decision Boundary')
    else:
        # If w2 is 0, the decision boundary is a vertical line at -bias / w1
        x_boundary = -bias / w1
        plt.axvline(x=x_boundary, color='green', linestyle='--', label='Decision Boundary')

    # Add labels and a legend
    plt.title('Input Data Visualization with Decision Boundary')
    plt.xlabel('Input 1 (x1)')
    plt.ylabel('Input 2 (x2)')
    plt.legend()

    # Display the plot
    plt.show()
if __name__ == "__main__":
    # Load input data from JSON file
    input_data = load_input_data('data.json')

    # Define the best weights and bias from the computation
    weights, bias = load_weights('weights.json')


    # Visualize the input data and the decision boundary
    visualize_input_data(input_data, weights, bias)