import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_and_analyze_limited_dataset(input_csv, output_csv, min_occurrences=2, max_occurrences=50):
    # Load the original CSV file
    df = pd.read_csv(input_csv)
    
    # Filter entities with at least min_occurrences occurrences
    filtered_df = df.groupby('Entity Name').filter(lambda x: len(x) >= min_occurrences)
    
    # Limit to max_occurrences for each entity
    limited_df = filtered_df.groupby('Entity Name').head(max_occurrences)
    
    # Save the limited DataFrame to a new CSV file
    limited_df.to_csv(output_csv, index=False)
    print(f"Modified dataset saved to {output_csv}")

    # Analyze the modified dataset
    # Calculate the frequency of each unique value in the "Entity Name" column
    entity_counts = limited_df['Entity Name'].value_counts()

    # Basic statistics
    print(f"Basic Statistics:\n{entity_counts.describe()}\n")
    
    # Histogram of entity occurrences
    plt.figure(figsize=(10, 6))
    sns.histplot(entity_counts, bins=30, kde=True)
    plt.title('Distribution of Entity Occurrences')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Frequency')
    plt.show()
    
    # Boxplot for outlier identification
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=entity_counts)
    plt.title('Boxplot of Entity Occurrences')
    plt.xlabel('Entity Occurrences')
    plt.show()
    
    # Outliers
    q1 = entity_counts.quantile(0.25)
    q3 = entity_counts.quantile(0.75)
    iqr = q3 - q1
    outlier_threshold_high = q3 + 1.5 * iqr
    outliers = entity_counts[entity_counts > outlier_threshold_high]
    print(f"Outliers (if any):\n{outliers}")

# Specify your input and output CSV file paths
input_csv_path = 'combined_books.csv'
output_csv_path = 'averaged_combined_books.csv'

# Create the modified dataset and analyze it
create_and_analyze_limited_dataset(input_csv_path, output_csv_path)
