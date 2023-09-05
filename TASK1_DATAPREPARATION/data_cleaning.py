import pandas as pd
import os
import sys

# Get the file name from command line argument
file_name = sys.argv[1]

# Create the path to the file
data_path = os.path.join(os.getcwd(), file_name)

# Load the data into a Pandas DataFrame
df = pd.read_csv(data_path, skiprows=3)  # Skip the first three rows

# Create new column names by joining the first three rows
new_headers = df.iloc[:3].apply(lambda x: '_'.join(x.dropna()), axis=0)
df.columns = new_headers

# Set the csv index and file name as the DataFrame index
df.set_index(["csv_index", "file_name"], inplace=True)

# Drop the first column as it doesn't contain useful information
df.drop(df.columns[0], axis=1, inplace=True)

# Save the cleaned data to a new file
cleaned_data_path = os.path.join(os.pardir, "cleaned_data", f"cleaned_{file_name}")
df.to_csv(cleaned_data_path)
