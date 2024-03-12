import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the CSV file
df = pd.read_csv('combined_books.csv')  # Make sure to replace 'combined_books.csv' with the path to your file

# Step 2: Filter based on the "Entity Name" column
entity_counts = df['Entity Name'].value_counts()
filtered_entities = entity_counts[(entity_counts > 5) & (entity_counts < 47)].index
filtered_df = df[df['Entity Name'].isin(filtered_entities)]

# Step 3: Write the filtered DataFrame to a new CSV file
filtered_df.to_csv('simple_averaged_dataset.csv', index=False)

# Step 4: Generate Visualizations
plt.figure(figsize=(12, 6))

# Original Data Histogram
sns.histplot(entity_counts, binwidth=1, kde=True, label='Original', color='blue')
plt.title('Histogram of Entity Occurrences: Original')
plt.xlabel('Number of Occurrences')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Filtered Data Histogram
filtered_entity_counts = filtered_df['Entity Name'].value_counts()
if not filtered_entity_counts.empty:
    plt.figure(figsize=(12, 6))
    sns.histplot(filtered_entity_counts, binwidth=1, kde=True, color='red', label='Filtered')
    plt.title('Histogram of Entity Occurrences: Filtered')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
else:
    print("No entities in the filtered dataset meet the histogram plotting criteria.")

# Box Plot Comparing the Number of Entities per Row Before and After Filtering
plt.figure(figsize=(10, 5))
# Prepare data
original_counts = df['Entity Name'].map(df['Entity Name'].value_counts())
filtered_counts = filtered_df['Entity Name'].map(filtered_df['Entity Name'].value_counts())
sns.boxplot(data=[original_counts, filtered_counts], palette=['blue', 'red'])
plt.xticks([0, 1], ['Original', 'Filtered'])
plt.title('Box Plot of Entity Counts per Entity')
plt.ylabel('Number of Entities')
plt.show()
