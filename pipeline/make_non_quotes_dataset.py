import pandas as pd
import re
import glob
import os
from tqdm import tqdm  # Import tqdm




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
    # Ensure text is a string
    if not isinstance(text, str):
        print(f"Warning: Non-string value encountered. Original type was {type(text)}. Converting to string.")
        text = str(text)

    
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






def clean_text(text):
    corrected_text = text.replace(" n't", "n't").replace(" n’", "n’").replace("( ", "(").replace(" ,", ",").replace("gon na", "gonna").replace(" n’t", "n’t").replace("“ ", "“")
    corrected_text = re.sub(r' (?=[^a-zA-Z0-9\s])', '', corrected_text)
    return corrected_text
'''
def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)
'''
import spacy

# Load the SpaCy language model
nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


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





import pandas as pd
import re
from tqdm import tqdm

def clean_text(text):
    corrected_text = text.replace(" n't", "n't").replace(" n’", "n’").replace("( ", "(").replace(" ,", ",").replace("gon na", "gonna").replace(" n’t", "n’t").replace("“ ", "“")
    corrected_text = re.sub(r' (?=[^a-zA-Z0-9\s])', '', corrected_text)
    return corrected_text

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
            nonquotes_data.append([words_chunk, last_end_id, start_id, "False", "Narrator", non_quote_context])
        
        last_end_id = end_id

    nonquotes_df = pd.DataFrame(nonquotes_data, columns=["Text", "Start Location", "End Location", "Is Quote", "Speaker", "Context"])
    nonquotes_df.to_csv(output_filePath, index=False)
    print(f"Saved nonquotes.csv to {output_filePath}")






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
        # Ensure text is a string, useful if the data includes NaNs or other non-string types
        text = str(text)

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


