import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import json

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


def load_data(path):
    pass
    # Load dataset

def prepare_data(df):
    pass
    # Prepare features and labels

    # Train-test-validation split

def normalize_features(data):
    scaler = StandardScaler()    
    normalized_data = scaler.fit_transform(data)
    return normalized_data
    # Normalize the features


# hiperparameters
def read_hyperparams(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)   


def main():
    epochs, brach_size, kernel_size = 20, 32, 9
    
    input_shape = (28, 28, 1)  # Example input shape for grayscale image data
    model = build_model_hardcoded("hyperparams.json", input_shape=input_shape)

    model.summary()
   # model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Compile the model
    # Train the model
    # Evaluate the model

if __name__ == "__main__":
    main()