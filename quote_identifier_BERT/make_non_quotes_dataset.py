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

def process_files(quotes_file, tokens_file):
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
    output_filename = os.path.join(os.path.dirname(quotes_file), "non_quotes.csv")
    nonquotes_df.to_csv(output_filename, index=False)
    print(f"Saved nonquotes.csv to {output_filename}")

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
