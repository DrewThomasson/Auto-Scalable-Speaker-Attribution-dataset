import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace 'your_file.csv' with the path to your actual CSV file
file_path = 'your_file.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Calculate the frequency of each unique value in the "Entity Name" column
entity_counts = df['Entity Name'].value_counts()

# Calculate the average occurrence of each entity
average_occurrence = entity_counts.mean()

# Identify outliers (for simplicity, here outliers are defined as entities with occurrences significantly higher or lower than average)
outliers = entity_counts[entity_counts > average_occurrence * 1.5]

# Visualize the distribution of entity occurrences
plt.figure(figsize=(10, 6))
sns.histplot(entity_counts, bins=30, kde=True)
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
