import pandas as pd

print("Starting feature engineering...")

# Correct the file paths if needed
file1 = 'data/customer_data1.csv'
file2 = 'data/customer_data2.csv'

print("Loading data...")

# Load the CSV files
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# Perform your feature engineering steps here
# For example, merging datasets or creating new features
processed_data = pd.merge(data1, data2, on='customer_id')

# Save the processed data
processed_data.to_csv('data/processed_data.csv', index=False)

print("Feature engineering completed. Data saved to 'data/processed_data.csv'")

