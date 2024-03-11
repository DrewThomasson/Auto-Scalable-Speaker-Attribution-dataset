


from booknlp.booknlp import BookNLP
import pandas as pd
import glob
import os
import nltk
import re
# Assume clean_text and process_files functions are defined here as previously described

def clean_text(text):
    corrected_text = text.replace(" n't", "n't").replace(" n’", "n’").replace("( ", "(").replace(" ,", ",").replace("gon na", "gonna").replace(" n’t", "n’t")
    corrected_text = re.sub(r' (?=[^a-zA-Z0-9\s])', '', corrected_text)
    return corrected_text

def process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=True):
    # Load the files
    df_quotes = pd.read_csv(quotes_file, delimiter="\t")
    df_entities = pd.read_csv(entities_file, delimiter="\t")
    df_tokens = pd.read_csv(tokens_file, delimiter="\t")

    character_info = {}

    def is_pronoun(word):
        tagged_word = nltk.pos_tag([word])
        return 'PRP' in tagged_word[0][1] or 'PRP$' in tagged_word[0][1]

    def get_gender(pronoun):
        male_pronouns = ['he', 'him', 'his']
        female_pronouns = ['she', 'her', 'hers']

        if pronoun.lower() in male_pronouns:
            return 'Male'
        elif pronoun.lower() in female_pronouns:
            return 'Female'
        return 'Unknown'

    # Process the quotes dataframe
    for index, row in df_quotes.iterrows():
        char_id = row['char_id']
        mention = row['mention_phrase']

        # Initialize character info if not already present
        if char_id not in character_info:
            character_info[char_id] = {"names": {}, "pronouns": {}, "quote_count": 0}

        # Update names or pronouns based on the mention_phrase
        if is_pronoun(mention):
            character_info[char_id]["pronouns"].setdefault(mention.lower(), 0)
            character_info[char_id]["pronouns"][mention.lower()] += 1
        else:
            character_info[char_id]["names"].setdefault(mention, 0)
            character_info[char_id]["names"][mention] += 1

        character_info[char_id]["quote_count"] += 1

    # Process the entities dataframe
    for index, row in df_entities.iterrows():
        coref = row['COREF']
        name = row['text']

        if coref in character_info:
            if is_pronoun(name):
                character_info[coref]["pronouns"].setdefault(name.lower(), 0)
                character_info[coref]["pronouns"][name.lower()] += 1
            else:
                character_info[coref]["names"].setdefault(name, 0)
                character_info[coref]["names"][name] += 1

    # Extract the most likely name and gender for each character
    for char_id, info in character_info.items():
        most_likely_name = max(info["names"].items(), key=lambda x: x[1])[0] if info["names"] else "Unknown"
        most_common_pronoun = max(info["pronouns"].items(), key=lambda x: x[1])[0] if info["pronouns"] else None

        gender = get_gender(most_common_pronoun) if most_common_pronoun else 'Unknown'
        gender_suffix = ".M" if gender == 'Male' else ".F" if gender == 'Female' else ".?"

        info["formatted_speaker"] = f"{char_id}:{most_likely_name}{gender_suffix}"
        info["most_likely_name"] = most_likely_name
        info["gender"] = gender

    def extract_surrounding_text(quote_start, quote_end, buffer=200):
        start_index = max(0, quote_start - buffer)
        end_index = quote_end + buffer
        surrounding_tokens = df_tokens[(df_tokens['token_ID_within_document'] >= start_index) & (df_tokens['token_ID_within_document'] <= end_index)]
        words_chunk = ' '.join([str(token_row['word']) for index, token_row in surrounding_tokens.iterrows()])
        return clean_text(words_chunk)

    # Write the formatted data to quotes_modified.csv, including text cleanup
    output_filename = os.path.join(os.path.dirname(quotes_file), "quotes_modified.csv")
    with open(output_filename, 'w', newline='') as outfile:
        fieldnames = ["Text", "Start Location", "End Location", "Is Quote", "Speaker", "Text Quote Is Contained In", "Entity Name"]
        writer = pd.DataFrame(columns=fieldnames)

        for index, row in df_quotes.iterrows():
            char_id = row['char_id']

            if not re.search('[a-zA-Z0-9]', row['quote']):
                print(f"Removing row with text: {row['quote']}")
                continue

            if character_info[char_id]["quote_count"] == 1:
                formatted_speaker = "Narrator"
                entity_name = "Narrator"
            else:
                formatted_speaker = character_info[char_id]["formatted_speaker"] if char_id in character_info else "Unknown"
                entity_name_parts = formatted_speaker.split(":")
                if len(entity_name_parts) > 1:
                    entity_name = entity_name_parts[1].split(".")[0]
                else:
                    entity_name = "Unknown"

            if not IncludeUnknownNames and entity_name == "Unknown":
                continue

            clean_quote_text = clean_text(row['quote'])
            surrounding_text = extract_surrounding_text(row['quote_start'], row['quote_end'])

            new_row = {
                "Text": clean_quote_text,
                "Start Location": row['quote_start'],
                "End Location": row['quote_end'],
                "Is Quote": "True",
                "Speaker": formatted_speaker,
                "Text Quote Is Contained In": surrounding_text,
                "Entity Name": entity_name
            }
            new_row_df = pd.DataFrame([new_row])
            writer = pd.concat([writer, new_row_df], ignore_index=True)

        return writer  # This returns the DataFrame instead of saving it
        print(f"Saved quotes_modified.csv to {output_filename}")
def process_book_with_booknlp(input_file, output_directory, book_id):
    model_params = {
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "big"
    }

    booknlp = BookNLP("en", model_params)
    booknlp.process(input_file, output_directory, book_id)

def process_all_books(input_dir, output_base_dir, include_unknown_names=True):
    all_book_files = glob.glob(os.path.join(input_dir, '*.txt'))
    combined_df = pd.DataFrame()

    for book_file in all_book_files:
        book_id = os.path.splitext(os.path.basename(book_file))[0]
        output_directory = os.path.join(output_base_dir, book_id)
        os.makedirs(output_directory, exist_ok=True)

        process_book_with_booknlp(book_file, output_directory, book_id)

        quotes_file = os.path.join(output_directory, f"{book_id}.quotes")
        entities_file = os.path.join(output_directory, f"{book_id}.entities")
        tokens_file = os.path.join(output_directory, f"{book_id}.tokens")
        
        if os.path.exists(quotes_file) and os.path.exists(entities_file) and os.path.exists(tokens_file):
            df = process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=include_unknown_names)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_csv_path = os.path.join(output_base_dir, "combined_books.csv")
    if not combined_df.empty:
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved to {combined_csv_path}")

def main():
    input_dir = 'books'
    output_base_dir = 'new_output_dir'
    process_all_books(input_dir, output_base_dir, include_unknown_names=False)

if __name__ == "__main__":
    main()
