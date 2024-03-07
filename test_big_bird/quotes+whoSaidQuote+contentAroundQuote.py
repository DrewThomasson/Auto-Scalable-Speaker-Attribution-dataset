import pandas as pd
import os
import glob
import re

def is_pronoun(word):
    # Define pronouns for simple identification (This could be expanded)
    pronouns = ['he', 'him', 'his', 'she', 'her', 'hers', 'I', 'me', 'my', 'we', 'us', 'our']
    return word.lower() in pronouns

def get_speaker_name(df_entities, char_id):
    # Filter entities by char_id (COREF)
    character_entities = df_entities[df_entities['COREF'] == char_id]

    # If direct name mentions are present, prefer those over pronouns
    direct_names = character_entities[~character_entities['text'].apply(is_pronoun)]
    if not direct_names.empty:
        # Return the most frequently mentioned name
        return direct_names['text'].value_counts().idxmax()
    else:
        # Fall back to pronouns if no direct names are available
        pronouns = character_entities['text'].value_counts()
        if not pronouns.empty:
            return pronouns.idxmax()  # Return the most frequently used pronoun
        else:
            return "Unknown"  # In case no entity matches

def process_files(quotes_file, tokens_file, entities_file):
    # Load the files
    df_quotes = pd.read_csv(quotes_file, delimiter="\t")
    df_tokens = pd.read_csv(tokens_file, delimiter="\t", on_bad_lines='skip', quoting=3)
    df_entities = pd.read_csv(entities_file, delimiter="\t")

    # Iterate through the quotes dataframe
    for index, row in df_quotes.iterrows():
        quote_start = row['quote_start']
        quote_end = row['quote_end']
        char_id = row['char_id']

        # Determine speaker
        speaker_name = get_speaker_name(df_entities, char_id)

        # Calculate start and end positions for the chunk of text around the quote
        chunk_start = max(0, quote_start - 200)  # Ensure start is not negative
        chunk_end = quote_end + 200  # End position 200 tokens after the quote

        # Filter tokens to get the chunk of text around the quote
        filtered_chunk = df_tokens[(df_tokens['token_ID_within_document'] >= chunk_start) & 
                                   (df_tokens['token_ID_within_document'] <= chunk_end)]
        
        # Build the text chunk including the quote
        text_chunk = ' '.join([str(token_row['word']) for _, token_row in filtered_chunk.iterrows()])
        text_chunk = text_chunk.replace(" n't", "n't").replace(" n’", "n’").replace(" ’", "’").replace(" ,", ",").replace(" .", ".").replace(" n’t", "n’t")
        text_chunk = re.sub(r' (?=[^a-zA-Z0-9\s])', '', text_chunk)
        
        # Print the chunk of text around the quote along with the speaker
        if text_chunk:
            print(f"Speaker: {speaker_name}\nQuote: {row['quote']}\nText Chunk (200 tokens before and after):\n{text_chunk}\n\n")

def main():
    # Specify the paths to your files
    quotes_file = 'g_book/g_book.quotes'
    tokens_file = 'g_book/g_book.tokens'
    entities_file = 'g_book/g_book.entities'
    
    # Process the files
    process_files(quotes_file, tokens_file, entities_file)

    print("All processing complete!")

if __name__ == "__main__":
    main()
