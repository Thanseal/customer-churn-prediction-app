import pandas as pd

def load_data(file_path):
    """Load the data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Perform basic data cleaning."""
    df = df.dropna()  # Drop missing values
    df = df.drop_duplicates()  # Drop duplicate rows
    return df

def save_processed_data(df, file_path):
    """Save the processed data to a CSV file."""
    df.to_csv(file_path, index=False)
