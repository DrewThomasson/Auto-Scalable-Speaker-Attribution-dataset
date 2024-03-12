import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import numpy as np

# Initialize tqdm for pandas operations
tqdm.pandas()

# Replace 'your_file.csv' with the path to your actual CSV file
file_path = 'your_file.csv'

# Define a function to read the CSV with a progress bar
def read_csv_with_progress(csv_path):
    # Get the number of lines in the CSV file
    total_lines = sum(1 for _ in open(csv_path, 'r', encoding='utf-8'))
    
    # Create a tqdm iterator for pandas
    tqdm_iterator = tqdm(total=total_lines, desc="Reading CSV")
    
    # Read the CSV file with the progress bar
    df = pd.read_csv(csv_path, iterator=True, chunksize=1000)
    df = pd.concat([chunk for chunk in df], ignore_index=True)
    
    # Close the tqdm iterator
    tqdm_iterator.close()
    
    return df

# Read the CSV file with progress bar
df = read_csv_with_progress(file_path)

# Calculate the frequency of each unique value in the "Entity Name" column
entity_counts = df['Entity Name'].value_counts()

# Print the occurrences for each entity from most common to least common
print("Occurrences for each entity (from most common to least common):")
print(entity_counts)

# Calculate the average occurrence of each entity
average_occurrence = entity_counts.mean()

# Identify outliers (for simplicity, here outliers are defined as entities with occurrences significantly higher or lower than average)
outliers = entity_counts[entity_counts > average_occurrence * 1.5]

# Visualize the distribution of entity occurrences for all unique values
plt.figure(figsize=(10, 6))
sns.histplot(entity_counts, bins=min(30, len(entity_counts)), kde=True)
plt.title('Distribution of Entity Occurrences')
plt.xlabel('Number of Occurrences')
plt.ylabel('Frequency')
plt.show()

# Boxplot to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=entity_counts)
plt.title('Boxplot of Entity Occurrences')
plt.xlabel('Entity Occurrences')
plt.show()

# Print the average occurrence
print(f'Average occurrence of entities: {average_occurrence}')

# Optional: Print outliers
print('Outliers:')
print(outliers)
