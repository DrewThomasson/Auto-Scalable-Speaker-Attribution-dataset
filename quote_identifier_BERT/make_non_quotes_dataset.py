import pandas as pd
import re
import glob
import os
from tqdm import tqdm  # Import tqdm

def clean_text(text):
    corrected_text = text.replace(" n't", "n't").replace(" n’", "n’").replace("( ", "(").replace(" ,", ",").replace("gon na", "gonna").replace(" n’t", "n’t").replace("“ ", "“")
    corrected_text = re.sub(r' (?=[^a-zA-Z0-9\s])', '', corrected_text)
    return corrected_text

def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def process_files(quotes_file, tokens_file, output_filePath):
    skip_rows = []
    while True:
        try:
            df_quotes = pd.read_csv(quotes_file, delimiter="\t", skiprows=skip_rows)
            break
        except pd.errors.ParserError as e:
            msg = str(e)
            match = re.search(r'at row (\d+)', msg)
            if match:
                problematic_row = int(match.group(1))
                print(f"Skipping problematic row {problematic_row} in {quotes_file}")
                skip_rows.append(problematic_row)
            else:
                print(f"Error reading {quotes_file}: {e}")
                return

    df_tokens = pd.read_csv(tokens_file, delimiter="\t", on_bad_lines='skip', quoting=3)
    
    last_end_id = 0
    nonquotes_data = []

    for index, row in tqdm(df_quotes.iterrows(), total=df_quotes.shape[0], desc="Processing Quotes"):
        start_id = row['quote_start']
        end_id = row['quote_end']
        
        expanded_tokens = df_tokens[(df_tokens['token_ID_within_document'] > last_end_id - 150) & 
                                    (df_tokens['token_ID_within_document'] < end_id + 150)]
        non_quote_context = ' '.join([str(token_row['word']) for index, token_row in expanded_tokens.iterrows()])
        non_quote_context = clean_text(non_quote_context)  # Apply text cleaning

        filtered_tokens = df_tokens[(df_tokens['token_ID_within_document'] > last_end_id) & 
                                    (df_tokens['token_ID_within_document'] < start_id)]
        words_chunk = ' '.join([str(token_row['word']) for index, token_row in filtered_tokens.iterrows()])
        words_chunk = clean_text(words_chunk)  # Apply text cleaning
        
        if words_chunk:
            sentences = split_into_sentences(words_chunk)
            for sentence in sentences:
                nonquotes_data.append([sentence, last_end_id, start_id, "False", "Narrator", non_quote_context])
        
        last_end_id = end_id

    nonquotes_df = pd.DataFrame(nonquotes_data, columns=["Text", "Start Location", "End Location", "Is Quote", "Speaker", "Context"])
    #output_filename = os.path.join(os.path.dirname(quotes_file), output_filePath)
    nonquotes_df.to_csv(output_filePath, index=False)
    print(f"Saved nonquotes.csv to {output_filePath}")




"""
def main():
    quotes_files = glob.glob('*.quotes', recursive=True)
    tokens_files = glob.glob('*.tokens', recursive=True)

    print("Starting processing...")
    for q_file in tqdm(quotes_files, desc="Overall Progress"):
        base_name = os.path.splitext(os.path.basename(q_file))[0]
        matching_token_files = [t_file for t_file in tokens_files if os.path.splitext(os.path.basename(t_file))[0] == base_name]

        if matching_token_files:
            process_files(q_file, matching_token_files[0])

    print("All processing complete!")

if __name__ == "__main__":
    main()



print("Removing any quotation marks from dataset...")
import pandas as pd
import re

# Load the CSV file
file_path = 'nonquotes.csv'  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Define a function to remove special characters
def remove_special_characters(text):
    # This regex matches any character that is NOT a letter, number, or space
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Apply the function to the "Context" and "Text" columns
df['Context'] = df['Context'].apply(remove_special_characters)
df['Text'] = df['Text'].apply(remove_special_characters)

# Save the cleaned data back to a CSV file
output_file_path = 'nonquotes.csv'  # Change this to your desired output file path
df.to_csv(output_file_path, index=False)

print("Special characters removed and data saved to", output_file_path)

"""


import pandas as pd
import re

def clean_csv_special_characters(input_file_path, output_file_path, columns):
    """
    Removes special characters from specified columns in a CSV file and saves the cleaned data.

    Parameters:
    - input_file_path: Path to the input CSV file.
    - output_file_path: Path where the cleaned CSV file will be saved.
    - columns: List of column names to clean.

    Returns:
    - None
    """

    # Load the CSV file
    df = pd.read_csv(input_file_path)

    # Define a function to remove special characters
    def remove_special_characters(text):
        # This regex matches any character that is NOT a letter, number, or space
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Apply the function to specified columns
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(remove_special_characters)
        else:
            print(f"Column {column} not found in the file.")

    # Save the cleaned data back to a CSV file
    df.to_csv(output_file_path, index=False)

    print(f"Special characters removed and data saved to {output_file_path}")
"""
# Example usage:
input_file_path = 'nonquotes.csv'  # Replace with your input file path
output_file_path = 'nonquotes.csv'  # Replace with your desired output file path
columns_to_clean = ['Context', 'Text']  # Columns to clean

clean_csv_special_characters(input_file_path, output_file_path, columns_to_clean)
"""