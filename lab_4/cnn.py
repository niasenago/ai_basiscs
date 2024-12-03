import argparse
import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

def escape_latex(s):
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('$', r'\$') \
            .replace('#', r'\#').replace('{', r'\{').replace('}', r'\}').replace('~', r'\~') \
            .replace('^', r'\^').replace('\'', r"\'").replace('"', r'\"')

def generate_latex_table(results):
    latex_table = """
\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|l|l|}
\\hline
\\textbf{Image Name} & \\textbf{Predicted Class} & \\textbf{Actual Class} \\\\
\\hline
"""
    for result in results:
        image_name, predicted_class, actual_class = result
        # Escape LaTeX special characters in the image name and class
        image_name = escape_latex(image_name)
        predicted_class = escape_latex(predicted_class)
        actual_class = escape_latex(actual_class)
        
        latex_table += f"{image_name} & {predicted_class} & {actual_class} \\\\ \n"
    
    latex_table += """
\\hline
\\end{tabular}
\\caption{Image Classification Results}
\\end{table}
"""
    return latex_table
def main():
    parser = argparse.ArgumentParser(description='Classify images and output results')
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('num_samples', type=int, help='Number of samples to classify')
    parser.add_argument('--latex', action='store_true', help='Generate output as LaTeX table')
    args = parser.parse_args()

    model = load_model(args.model_path)

    classes = sorted(os.listdir(args.data_dir))
    print(f"Detected classes: {classes}")
    
    all_samples = []
    for class_name in classes:
        class_dir = os.path.join(args.data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        all_samples.extend([(os.path.join(class_dir, f), class_name) for f in image_files])
    
    random_samples = random.sample(all_samples, args.num_samples)

    results = []
    for image_path, actual_class in random_samples:
        img = preprocess_image(image_path)
        
        # Predict the class
        prediction = model.predict(img, verbose=0)[0][0]
        predicted_class = classes[0] if prediction < 0.5 else classes[1]
        
        # Print the result
        image_name = os.path.basename(image_path)
        results.append((image_name, predicted_class, actual_class))
        print(f"{image_name:<30} {predicted_class:<20} {actual_class:<20}")

    # If --latex flag is set, generate and print the LaTeX table
    if args.latex:
        latex_table = generate_latex_table(results)
        print("\nGenerated LaTeX Table:\n")
        print(latex_table)

if __name__ == '__main__':
    main()