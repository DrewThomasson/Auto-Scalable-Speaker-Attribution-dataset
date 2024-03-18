
from booknlp.booknlp import BookNLP
import pandas as pd
import glob
import os
from tqdm import tqdm

# Import your dataset creation scripts
from make_non_quotes_dataset import clean_text, split_into_sentences, process_files as process_non_quotes_files, clean_csv_special_characters
from make_quotes_dataset import process_files as process_quotes_files, clean_csv, split_sentences_and_expand_csv

# Download necessary NLTK data
import nltk
nltk.download('punkt')

def process_book(input_file, output_directory, book_id):
    model_params={
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "big"
    }

    booknlp = BookNLP("en", model_params)
    #booknlp.process(input_file, output_directory, book_id)

def process_all_books(input_dir='books/'):
    for input_file in glob.glob(os.path.join(input_dir, '*.txt')):
        book_name = os.path.splitext(os.path.basename(input_file))[0]
        output_directory = f'output_dir/{book_name}/'
        os.makedirs(output_directory, exist_ok=True)
        
        # Process the book with BookNLP to generate .quotes, .tokens, and .entities files
        process_book(input_file, output_directory, book_name)
        
        # Find the generated files
        quotes_file_path = glob.glob(os.path.join(output_directory, f'{book_name}.quotes'))[0]
        tokens_file_path = glob.glob(os.path.join(output_directory, f'{book_name}.tokens'))[0]
        entities_file_path = glob.glob(os.path.join(output_directory, f'{book_name}.entities'))[0]
        
        # Assuming your process_files functions are correctly defined and work with the paths
        # Generate non_quotes.csv for the current book
        non_quotes_output_filePath = os.path.join(output_directory, f'{book_name}_non_quotes.csv')
        ##process_non_quotes_files(quotes_file_path, tokens_file_path, non_quotes_output_filePath)  # Pass correct file paths
        from make_non_quotes_dataset import clean_text
        from make_non_quotes_dataset import split_into_sentences
        from make_non_quotes_dataset import process_files
        from make_non_quotes_dataset import clean_csv_special_characters
        process_files(quotes_file_path, tokens_file_path, non_quotes_output_filePath)
        #from make_quotes_dataset import clean_csv_special_characters
        from make_quotes_dataset import split_csv_by_pauses
        #split_csv_by_pauses(non_quotes_output_filePath, non_quotes_output_filePath)
        columns_to_clean = ['Context', 'Text']  # Columns to clean
        clean_csv_special_characters(non_quotes_output_filePath, non_quotes_output_filePath, columns_to_clean)
        
        # Generate quotes.csv for the current book
        quotes_output_filePath = os.path.join(output_directory, f'{book_name}_quotes.csv')
        ##process_quotes_files(quotes_file_path, entities_file_path, tokens_file_path, quotes_output_filePath)  # Pass correct file paths
        from make_quotes_dataset import clean_text
        from make_quotes_dataset import process_files
        output_file_path = quotes_output_filePath
        process_files(quotes_file_path, entities_file_path, tokens_file_path, IncludeUnknownNames=True, output_file_path=quotes_output_filePath)
        from make_quotes_dataset import split_csv_by_pauses
        #split_csv_by_pauses(quotes_output_filePath, quotes_output_filePath)
        columns_to_clean = ['Context', 'Text']  # Columns to clean
        clean_csv_special_characters(quotes_output_filePath, quotes_output_filePath, columns_to_clean)




process_all_books()


import pandas as pd
import glob
import os

import pandas as pd
import os

def combine_csv_files(root_dir):
    all_csv_files = []

    # Walk through all directories and files starting from root_dir
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file is a CSV
            if file.endswith('.csv'):
                all_csv_files.append(os.path.join(subdir, file))

    # Define the required columns
    required_columns = ["Text", "Start Location", "End Location", "Is Quote", "Speaker", "Context"]
    optional_columns = ["Entity Name"]

    # Initialize an empty DataFrame to hold combined data
    combined_df = pd.DataFrame()

    # Process each CSV file
    for csv_file in all_csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Ensure all required columns are present, fill missing ones with NA
            for col in required_columns:
                if col not in df.columns:
                    df[col] = "NA"

            # Check and add the optional column if missing
            for col in optional_columns:
                if col not in df.columns:
                    df[col] = "Probs Narrator"

            # Append to the combined DataFrame
            combined_df = pd.concat([combined_df, df[required_columns + optional_columns]], ignore_index=True)
        
        except pd.errors.EmptyDataError:
            print(f"Skipping {csv_file} due to being empty or improperly formatted.")

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv('combined_csv.csv', index=False)

    return combined_df

# Example usage
# Replace 'path_to_your_directory' with the actual path to the directory containing your CSV files
root_dir = "output_dir"
combined_data = combine_csv_files(root_dir)

