import random
import numpy as np
import pandas as pd

def read_data(file_path, delimiter=","):
    data = pd.read_csv(file_path, header=None, delimiter=delimiter)
    
    # Get the number of attributes (columns) based on the first row
    num_attributes = data.shape[1] - 1  # The last column is the class label
    
    # Create dummy column names; 
    # todo:mb we don't need this
    column_names = [f'Attribute_{i+1}' for i in range(num_attributes)] + ['Class']
    
    data.columns = column_names
    return data


# after normalization number of elements in the each class should be greater or equal to 200
def normilize_data(dataframe, target_size=250, noise_level=0.01):
    augmented_data = []  # Initialize as an empty list instead of DataFrame
    
    # Iterate over each class in the dataset
    for class_label in dataframe['Class'].unique():
        class_data = dataframe[dataframe['Class'] == class_label].values
        current_size = len(class_data)
        
        # If class already has enough elements, no augmentation is needed
        if current_size >= target_size:
            augmented_class_data = class_data
        else:
            # Apply random sampling with noise to create additional samples
            num_samples_needed = target_size - current_size
            noisy_samples = random_sampling_with_noise(class_data, num_samples_needed, noise_level)
            augmented_class_data = np.vstack([class_data, noisy_samples])
        
        # Add the augmented class data to the list
        augmented_data.append(pd.DataFrame(augmented_class_data, columns=dataframe.columns))
    
    augmented_data = pd.concat(augmented_data, ignore_index=True)
    
    return augmented_data

# applies random sampling with gausian noise
def random_sampling_with_noise(dataset, num_samples, noise_level=0.01):
    noisy_samples = []
    num_attributes = len(dataset[0]) - 1  # exclude the class label from the attributes
    
    for _ in range(num_samples):
        random_sample = random.choice(dataset)
        noisy_sample = [value + np.random.normal(0, noise_level) for value in random_sample[:-1]]
        noisy_sample.append(int(random_sample[-1]))  # workaraund to save class as int value
        noisy_samples.append(noisy_sample)
    
    return noisy_samples

def shuffle_data(df):
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    return shuffled_df

def write_data(df, file_path, delimiter=","):
    dataframe = shuffle_data(df)
    # workaraund to save class as int value
    dataframe['Class'] = dataframe['Class'].astype(int)
    
    dataframe.to_csv(file_path, index=False, sep=delimiter, header=False)
    print(f"DataFrame saved to {file_path}")

def main():
    file_path = './data/iris.data' 
    df = read_data(file_path)
    augmented_df = normilize_data(df)
    write_data(augmented_df, 'data/iris.augmented.data')

if __name__ == "__main__":
   main()