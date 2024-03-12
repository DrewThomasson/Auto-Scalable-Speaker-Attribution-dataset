import pandas as pd

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

# Specify your input and output CSV file paths
input_csv_path = 'your_input_file.csv'
output_csv_path = 'final_dataset.csv'

# Adjust the dataset and save the cleaned version
adjust_and_clean_dataset(input_csv_path, output_csv_path)
