import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_dataset(title, entity_counts):
    """Generate visualizations for the given dataset."""
    plt.figure(figsize=(15, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(entity_counts, bins=30, kde=True)
    plt.title(f'Histogram of {title}')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Frequency')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=entity_counts)
    plt.title(f'Boxplot of {title}')
    plt.xlabel('Entity Occurrences')

    plt.tight_layout()
    plt.show()

def compare_datasets(input_csv, output_csv):
    """Load datasets, generate visualizations, and compare them."""
    # Load the original and final CSV files
    original_df = pd.read_csv(input_csv)
    final_df = pd.read_csv(output_csv)

    # Calculate the occurrences for both datasets
    original_counts = original_df['Entity Name'].value_counts()
    final_counts = final_df['Entity Name'].value_counts()

    # Visualize the original dataset
    visualize_dataset('Original Dataset', original_counts)

    # Visualize the final (adjusted) dataset
    visualize_dataset('Final Dataset', final_counts)

    # Comparison of basic statistics
    print("Comparison of Basic Statistics:")
    print("Original Dataset:\n", original_counts.describe())
    print("\nFinal Dataset:\n", final_counts.describe())

def adjust_and_clean_dataset(input_csv, output_csv):
    # Load the original CSV file
    df = pd.read_csv(input_csv)
    
    # Calculate the average number of occurrences per unique entity
    avg_occurrences = int(df['Entity Name'].value_counts().mean())
    
    # Limit to avg_occurrences for each entity
    limited_df = df.groupby('Entity Name').head(avg_occurrences)
    
    # Calculate the occurrences again to find outliers
    entity_counts = limited_df['Entity Name'].value_counts()
    
    # Compute Q1, Q3, and IQR for the limited dataset
    Q1 = entity_counts.quantile(0.25)
    Q3 = entity_counts.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify entities within the acceptable range (non-outliers)
    acceptable_entities = entity_counts[(entity_counts >= lower_bound) & (entity_counts <= upper_bound)].index
    
    # Keep only rows for acceptable entities
    final_df = limited_df[limited_df['Entity Name'].isin(acceptable_entities)]
    
    # Save the final DataFrame to a new CSV file
    final_df.to_csv(output_csv, index=False)
    print(f"Final dataset saved to {output_csv}. It includes entities limited to the average number of occurrences and without outliers.")

    # Analyze and compare the datasets
    compare_datasets(input_csv, output_csv)

# Specify your input and output CSV file paths
input_csv_path = 'combined_books.csv'
output_csv_path = 'IQR_combined_dataset.csv'

# Adjust the dataset and save the cleaned version, then analyze and compare
adjust_and_clean_dataset(input_csv_path, output_csv_path)
