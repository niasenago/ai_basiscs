import argparse
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="Evaluate a trained model and optionally export results.")
parser.add_argument("model_path", type=str, help="Path to the saved model (e.g., results/4_model/4_model.h5).")
parser.add_argument("test_data_dir", type=str, help="Path to the test data directory.")
parser.add_argument("--export", action="store_true", help="Flag to export the confusion matrix and accuracy to the output directory.")
args = parser.parse_args()

model = load_model(args.model_path)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    args.test_data_dir,
    target_size=(128, 128), 
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Čihuahua", "Keksiukai"], yticklabels=["Čihuahua", "Keksiukai"])
plt.title("Klasifikavimo matrica")
plt.xlabel("Nuspėta klasė")
plt.ylabel("Tikra klasė")

if args.export:

    output_dir = os.path.dirname(args.model_path)  
    os.makedirs(output_dir, exist_ok=True)
    
    conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    print(f"Confusion matrix saved at: {conf_matrix_path}")
    
else:
    plt.show()
