import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import logging
from tensorflow.keras.models import load_model
import json
import argparse
import io
from contextlib import redirect_stdout

def build_model_from_json(json_path, input_shape=None):
    with open(json_path, 'r') as file:
        config = json.load(file)
    
    model = Sequential()
    first_layer = True

    for layer in config.get("layers", []):
        layer_type = layer.get("type")
        
        if layer_type == "Conv2D":
            if first_layer and input_shape:
                model.add(Conv2D(filters=layer["filters"], kernel_size=tuple(layer["kernel_size"]),
                                 activation=layer["activation"], input_shape=input_shape))
            else:
                model.add(Conv2D(filters=layer["filters"], kernel_size=tuple(layer["kernel_size"]),
                                 activation=layer["activation"]))
        
        elif layer_type == "MaxPooling2D":
            model.add(MaxPooling2D(pool_size=tuple(layer["pool_size"])))

        elif layer_type == "Dropout":
            model.add(Dropout(rate=layer["rate"]))
        elif layer_type == "BatchNormalization":
            model.add(BatchNormalization())             

        first_layer = False  # Input shape only for the first layer

    # Flatten the output and add a final fully connected layer (optional for classification)
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))  # Final classification layer (for binary classification)

    # Compile the model with training parameters from the JSON
    training_config = config.get("training", {})
    optimizer = training_config.get("optimizer", "adam")
    loss_function = training_config.get("loss_function", "binary_crossentropy")
    metrics = ["accuracy"]  # Default metrics can be extended
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    return model

def build_model_hardcoded(json_path, input_shape):
    with open(json_path, 'r') as file:
        config = json.load(file)
    
    training_config = config.get("training", {})
    optimizer = training_config.get("optimizer", "adam") #default is adam
    loss_function = training_config.get("loss_function", "binary_crossentropy") # default is binary_crossentropy
    metrics = ["accuracy"]  

    # hardcoded architecture
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    ###
    # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time,
    # which helps prevent overfitting. 
    # Inputs not set to 0 are scaled up by 1 / (1 - rate) such that the sum over all inputs is unchanged.
    #

    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))  

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    return model



def load_pre_split_data(train_dir, val_dir, test_dir, target_size=(128, 128), batch_size=32):
    """
    Load already split data from pre-defined folders.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Normalize pixel values

    # Load training data
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True
    )

    # Load validation data
    val_data = datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    # Load test data
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    return train_data, val_data, test_data

def prepare_data(data, test_size=0.1, val_size=0.1):
    """
    Prepare features and labels and split into train, validation, and test sets.
    """
    X = data[0]
    y = data[1]

    # Split into train and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    
    # Split the temp set into validation and test
    test_split_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split_ratio, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_features(data):
    scaler = StandardScaler()    
    normalized_data = scaler.fit_transform(data)
    return normalized_data
    # Normalize the features (mb we should also devide each color by 255 )

def log_model_summary(model, log_file):
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        summary = buf.getvalue()
    with open(log_file, "a") as f:
        f.write("\nModel Architecture:\n")
        f.write(summary)
    logging.info("Model architecture logged successfully.")


def main():
    try:
        parser = argparse.ArgumentParser(description="Train a model based on a hyperparameters JSON file.")

        # Add arguments
        parser.add_argument(
            "--hyperparams",
            type=str,
            required=True,
            help="Path to the hyperparameters JSON file"
        )
        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="Name of the model"
        )

        # Parse arguments
        args = parser.parse_args()

        # Extract arguments
        hyperparams_path = args.hyperparams
        model_name = args.model_name  
        logging.basicConfig(
            filename=f"{model_name}_training.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"  # Overwrite existing log file
        )

        # Log starting of the program
        logging.info("Program started.")
        logging.info(f"Using hyperparameters file: {hyperparams_path}")
        logging.info(f"Your model name is: {model_name}")


        # Load hyperparameters
        with open(hyperparams_path, 'r') as file:
            hyperparams = json.load(file)
        epochs = hyperparams["training"]["epochs"]
        batch_size = hyperparams["training"]["batch_size"]
        logging.info(f"Hyperparameters loaded: epochs={epochs}, batch_size={batch_size}")

        input_shape = (128, 128, 3)  # Example input shape for RGB image data
        
        model = build_model_from_json(hyperparams_path, input_shape=input_shape)
        
        logging.info("Model built successfully.")
        log_model_summary(model, f"{model_name}.summary.log")

        # Load pre-split data
        train_dir = "./data/new_train"
        val_dir = "./data/new_val"
        test_dir = "./data/new_test"
        train_data, val_data, test_data = load_pre_split_data(train_dir, val_dir, test_dir)
        logging.info("Data loaded successfully from pre-split directories.")

        # Train the model
        logging.info("Starting training...")
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data
        )
        logging.info("Training completed.")
        
        history_path = f"{model_name}_history.json"
        with open(history_path, 'w') as file:
            json.dump(history.history, file)
        logging.info(f"Training history saved to '{history_path}'.")

        # Evaluate the model
        logging.info("Starting evaluation on the test data...")
        loss, accuracy = model.evaluate(test_data)
        logging.info(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

        # Save the trained model
        model.save(f"{model_name}.h5")
        logging.info(f"Model saved as '{model_name}.h5'.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()