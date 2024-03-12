import pandas as pd

def limit_rows_per_entity(input_csv, output_csv, max_rows_per_entity=50):
    # Load the original CSV file
    df = pd.read_csv(input_csv)
    
    # Function to limit rows of each group
    def limit_rows(group):
        return group.head(max_rows_per_entity)
    
    # Apply the function to each group and concatenate the results
    limited_df = df.groupby('Entity Name', group_keys=False).apply(limit_rows)
    
    # Save the limited DataFrame to a new CSV file
    limited_df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")

# Replace 'your_input_file.csv' with the path to your actual CSV file
input_csv_path = 'combined_books.csv'
# Specify the path for the new CSV file
output_csv_path = 'averaged_combined_books.csv'

# Call the function
limit_rows_per_entity(input_csv_path, output_csv_path)
