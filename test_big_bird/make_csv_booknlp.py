import os
import pandas as pd
import re
from booknlp.booknlp import BookNLP

# Global list to hold data from all books
all_books_data = []

def is_pronoun(word):
    pronouns = ['he', 'him', 'his', 'she', 'her', 'hers', 'I', 'me', 'my', 'we', 'us', 'our']
    return word.lower() in pronouns

def get_speaker_name(df_entities, char_id):
    character_entities = df_entities[df_entities['COREF'] == char_id]
    direct_names = character_entities[~character_entities['text'].apply(is_pronoun)]
    if not direct_names.empty:
        return direct_names['text'].value_counts().idxmax()
    else:
        pronouns = character_entities['text'].value_counts()
        if not pronouns.empty:
            return pronouns.idxmax()
        else:
            return "Unknown"

def process_files(quotes_file, tokens_file, entities_file):
    df_quotes = pd.read_csv(quotes_file, delimiter="\t")
    df_tokens = pd.read_csv(tokens_file, delimiter="\t", on_bad_lines='skip', quoting=3)
    df_entities = pd.read_csv(entities_file, delimiter="\t")

    for index, row in df_quotes.iterrows():
        quote_start = row['quote_start']
        quote_end = row['quote_end']
        char_id = row['char_id']

        speaker_name = get_speaker_name(df_entities, char_id)

        chunk_start = max(0, quote_start - 200)
        chunk_end = quote_end + 200

        filtered_chunk = df_tokens[(df_tokens['token_ID_within_document'] >= chunk_start) &
                                   (df_tokens['token_ID_within_document'] <= chunk_end)]

        text_chunk = ' '.join([str(token_row['word']) for _, token_row in filtered_chunk.iterrows()])
        text_chunk = text_chunk.replace(" n't", "n't").replace(" n’", "n’").replace(" ’", "’").replace(" ,", ",").replace(" .", ".").replace(" n’t", "n’t")
        text_chunk = re.sub(r' (?=[^a-zA-Z0-9\s])', '', text_chunk)

        all_books_data.append({
            "chunk_of_text": text_chunk,
            "specific_quote": row['quote'],
            "entity_who_said_quote": speaker_name
        })

def process_books(input_folder):
    model_params = {
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "big"
    }

    booknlp = BookNLP("en", model_params)

    if not input_folder.endswith('/'):
        input_folder += '/'

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            input_file = input_folder + file_name
            book_id = file_name.split('.')[0]
            output_directory = f"output_dir/{book_id}/"
            os.makedirs(output_directory, exist_ok=True)

            print(f"Processing {file_name}...")
            booknlp.process(input_file, output_directory, book_id)

            quotes_file = f"{output_directory}{book_id}.quotes"
            tokens_file = f"{output_directory}{book_id}.tokens"
            entities_file = f"{output_directory}{book_id}.entities"

            process_files(quotes_file, tokens_file, entities_file)

    # Generate the combined CSV with specified column names
    giant_csv_path = "output_dir/all_books_processed_data.csv"
    df_all_books = pd.DataFrame(all_books_data)
    df_all_books.to_csv(giant_csv_path, index=False)
    print(f"Combined data written to {giant_csv_path}")

if __name__ == "__main__":
    input_folder = "books"  # Change to your folder path
    process_books(input_folder)
