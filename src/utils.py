import os

def ensure_dir_exists(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def print_dataset_summary(df):
    """Print a summary of the dataset."""
    print("Dataset Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())