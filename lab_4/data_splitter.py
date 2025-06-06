import os
import shutil
from sklearn.model_selection import train_test_split
import random

# paths and desired ratios
data_dir = "./data"
all_data_dir = "./data/all_data"
output_dirs = {
    "train": "./data/new_train",
    "val": "./data/new_val",
    "test": "./data/new_test"
}
ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
reduce_factor = 0.7  

# Combine all data into one directory
def combine_data(data_dir, all_data_dir):
    if not os.path.exists(all_data_dir):
        os.makedirs(all_data_dir)
    
    for category in ["chihuahua", "muffin"]:
        category_dir = os.path.join(all_data_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for subdir in ["train", "test"]:
            source_dir = os.path.join(data_dir, subdir, category)
            if os.path.exists(source_dir):
                for file in os.listdir(source_dir):
                    source_file = os.path.join(source_dir, file)
                    dest_file = os.path.join(category_dir, file)
                    shutil.copy(source_file, dest_file)

# Reduce the number of samples
def reduce_samples(all_data_dir, reduce_factor):
    for category in ["chihuahua", "muffin"]:
        category_path = os.path.join(all_data_dir, category)
        files = os.listdir(category_path)
        
        # Calculate the reduced number of samples
        reduced_count = int(len(files) * reduce_factor)
        reduced_files = random.sample(files, reduced_count)
        
        # Clear the directory and keep only the reduced files
        for file in files:
            file_path = os.path.join(category_path, file)
            if file not in reduced_files:
                os.remove(file_path)

# Re-divide data into train, validation, and test sets
def split_data(all_data_dir, output_dirs, ratios):
    for category in ["chihuahua", "muffin"]:
        category_path = os.path.join(all_data_dir, category)
        files = os.listdir(category_path)
        random.shuffle(files)  # Ensure random splitting
        
        # Calculate split indices
        total_files = len(files)
        train_split = int(ratios["train"] * total_files)
        val_split = train_split + int(ratios["val"] * total_files)
        
        train_files = files[:train_split]
        val_files = files[train_split:val_split]
        test_files = files[val_split:]
        
        # Copy files to respective directories
        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_dir = os.path.join(output_dirs[split], category)
            os.makedirs(split_dir, exist_ok=True)
            for file in split_files:
                source_file = os.path.join(category_path, file)
                dest_file = os.path.join(split_dir, file)
                shutil.copy(source_file, dest_file)

if __name__ == "__main__":
    combine_data(data_dir, all_data_dir)
    
    reduce_samples(all_data_dir, reduce_factor)
    
    split_data(all_data_dir, output_dirs, ratios)
    
    print("Data preparation and splitting complete!")
