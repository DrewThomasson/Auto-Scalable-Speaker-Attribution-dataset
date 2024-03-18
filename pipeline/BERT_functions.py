#split csv by pauses

import pandas as pd
import glob
import os
import nltk
from nltk.tokenize import sent_tokenize
import re
import subprocess
import csv
from tqdm import tqdm

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt')



def replace_titles_and_abbreviations(text):
    # Dictionary of patterns and their replacements, using angle brackets for emphasis
    replacements = {
        r"Mr\.": "<MR>",
        r"Ms\.": "<MS>",
        r"Mrs\.": "<MRS>",
        r"Dr\.": "<DR>",
        r"Prof\.": "<PROF>",
        r"Rev\.": "<REV>",
        r"Gen\.": "<GEN>",
        r"Sen\.": "<SEN>",
        r"Rep\.": "<REP>",
        r"Gov\.": "<GOV>",
        r"Lt\.": "<LT>",
        r"Sgt\.": "<SGT>",
        r"Capt\.": "<CAPT>",
        r"Cmdr\.": "<CMDR>",
        r"Adm\.": "<ADM>",
        r"Maj\.": "<MAJ>",
        r"Col\.": "<COL>",
        r"St\.": "<ST>",  # Saint or Street
        r"Co\.": "<CO>",
        r"Inc\.": "<INC>",
        r"Corp\.": "<CORP>",
        r"Ltd\.": "<LTD>",
        r"Jr\.": "<JR>",
        r"Sr\.": "<SR>",
        r"Ph\.D\.": "<PHD>",
        r"M\.D\.": "<MD>",
        r"B\.A\.": "<BA>",
        r"B\.S\.": "<BS>",
        r"M\.A\.": "<MA>",
        r"M\.S\.": "<MS>",
        r"LL\.B\.": "<LLB>",
        r"LL\.M\.": "<LLM>",
        r"J\.D\.": "<JD>",
        r"Esq\.": "<ESQ>",
        # Add more replacements as needed
    }
    
    # Iterate through the dictionary and replace all occurrences
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text






def revert_titles_and_abbreviations(text):
    # Dictionary of custom placeholders back to original forms
    replacements = {
        r"<MR>": "Mr.",
        r"<MS>": "Ms.",
        r"<MRS>": "Mrs.",
        r"<DR>": "Dr.",
        r"<PROF>": "Prof.",
        r"<REV>": "Rev.",
        r"<GEN>": "Gen.",
        r"<SEN>": "Sen.",
        r"<REP>": "Rep.",
        r"<GOV>": "Gov.",
        r"<LT>": "Lt.",
        r"<SGT>": "Sgt.",
        r"<CAPT>": "Capt.",
        r"<CMDR>": "Cmdr.",
        r"<ADM>": "Adm.",
        r"<MAJ>": "Maj.",
        r"<COL>": "Col.",
        r"<ST>": "St.",  # Saint or Street, context needed
        r"<CO>": "Co.",
        r"<INC>": "Inc.",
        r"<CORP>": "Corp.",
        r"<LTD>": "Ltd.",
        r"<JR>": "Jr.",
        r"<SR>": "Sr.",
        r"<PHD>": "Ph.D.",
        r"<MD>": "M.D.",
        r"<BA>": "B.A.",
        r"<BS>": "B.S.",
        r"<MA>": "M.A.",
        r"<MS>": "M.S.",
        r"<LLB>": "LL.B.",
        r"<LLM>": "LL.M.",
        r"<JD>": "J.D.",
        r"<ESQ>": "Esq.",
        # Add more reversals as needed
    }
    
    # Iterate through the dictionary and replace all occurrences
    for placeholder, original in replacements.items():
        text = re.sub(placeholder, original, text)
    
    return text








def split_csv_by_pauses(input_csv_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Function to split text based on pauses
    def split_text(text):
        # Define the pattern for splitting: period, comma, exclamation, semicolon
        pattern = r'[.!,;?:]'
        # Split text and filter out any empty strings from the list
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        # Add the punctuation back to the parts except the last one
        parts = [part + text[text.find(part) + len(part)] if text.find(part) + len(part) < len(text) and text[text.find(part) + len(part)] in '.!,;?' else part for part in parts]
        return parts

    #change all the surnames into the codenames so it doesnt split at their points
    df['Text'] = df['Text'].apply(replace_titles_and_abbreviations)

    # Apply the function to split the "Text" column and explode into new rows
    df['Split_Text'] = df['Text'].apply(split_text)

    # Explode the 'Split_Text' column to create new rows
    exploded_df = df.explode('Split_Text')

    # Replace the original 'Text' column with the exploded 'Split_Text' column
    exploded_df['Text'] = exploded_df['Split_Text']
    exploded_df.drop('Split_Text', axis=1, inplace=True)

    # Reset index if necessary
    exploded_df.reset_index(drop=True, inplace=True)

    #Revert all of the surname codenames back into regular surname extrassions
    exploded_df['Text'] = exploded_df['Text'].apply(revert_titles_and_abbreviations)

    # Save the resulting DataFrame to the specified output CSV file
    exploded_df.to_csv(output_csv_path, index=False)

    print(f'Processed CSV file saved to {output_csv_path}')


#remove all special charcters from csv



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





#ex of use
"""        from make_quotes_dataset import split_csv_by_pauses
        split_csv_by_pauses(non_quotes_output_filePath, non_quotes_output_filePath)
        columns_to_clean = ['Context', 'Text']  # Columns to clean
        clean_csv_special_characters(non_quotes_output_filePath, non_quotes_output_filePath, columns_to_clean)"""
