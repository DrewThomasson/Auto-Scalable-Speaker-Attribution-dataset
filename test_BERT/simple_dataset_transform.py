import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the CSV file
df = pd.read_csv('combined_books.csv')  # replace 'your_file.csv' with the path to your file

# Step 2: Filter based on the "Entity Name" column
entity_counts = df['Entity Name'].value_counts()
filtered_entities = entity_counts[(entity_counts > 5) & (entity_counts < 47)].index
filtered_df = df[df['Entity Name'].isin(filtered_entities)]

# Step 3: Write the filtered DataFrame to a new CSV file
filtered_df.to_csv('simple_averaged_dataset.csv', index=False)

# Step 4: Generate Visualizations
# Histogram of entity occurrences before and after filtering
plt.figure(figsize=(10, 5))
sns.histplot(entity_counts, binwidth=1, kde=True, label='Original')
sns.histplot(filtered_entities.value_counts(), binwidth=1, kde=True, color='red', label='Filtered')
plt.title('Histogram of Entity Occurrences')
plt.xlabel('Number of Occurrences')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Box plot comparing the number of entities per row before and after filtering
plt.figure(figsize=(10, 5))
df['Entity Count'] = df['Entity Name'].map(df['Entity Name'].value_counts())
filtered_df['Entity Count'] = filtered_df['Entity Name'].map(filtered_df['Entity Name'].value_counts())
sns.boxplot(data=pd.DataFrame({'Original': df['Entity Count'], 'Filtered': filtered_df['Entity Count']}))
plt.title('Box Plot of Entity Counts per Row')
plt.ylabel('Number of Entities')
plt.show()

# You can add more visualizations as needed
