import pandas as pd
import os
import glob
import re

def process_files(quotes_file, tokens_file):
    # Load the files
    df_quotes = pd.read_csv(quotes_file, delimiter="\t")
    df_tokens = pd.read_csv(tokens_file, delimiter="\t", on_bad_lines='skip', quoting=3)

    # Iterate through the quotes dataframe
    for index, row in df_quotes.iterrows():
        quote_start = row['quote_start']
        quote_end = row['quote_end']
        
        # Calculate start and end positions for the chunk of text around the quote
        chunk_start = max(0, quote_start - 200)  # Ensure start is not negative
        chunk_end = quote_end + 200  # End position 100 tokens after the quote

        # Filter tokens to get the chunk of text around the quote
        filtered_chunk = df_tokens[(df_tokens['token_ID_within_document'] >= chunk_start) & 
                                   (df_tokens['token_ID_within_document'] <= chunk_end)]
        
        # Build the text chunk including the quote
        text_chunk = ' '.join([str(token_row['word']) for _, token_row in filtered_chunk.iterrows()])
        text_chunk = text_chunk.replace(" n't", "n't").replace(" n’", "n’").replace(" ’", "’").replace(" ,", ",").replace(" .", ".").replace(" n’t", "n’t")
        text_chunk = re.sub(r' (?=[^a-zA-Z0-9\s])', '', text_chunk)
        
        # Print the chunk of text around the quote
        if text_chunk:
            print(f"Quote: {row['quote']}\nText Chunk (100 tokens before and after):\n{text_chunk}\n\n")

def main():
    # Specify the paths to your quotes and tokens files
    quotes_file = 'g_book/g_book.quotes'
    tokens_file = 'g_book/g_book.tokens'
    
    # Process the files
    process_files(quotes_file, tokens_file)

    print("All processing complete!")

if __name__ == "__main__":
    main()
