import pandas as pd
import re
import glob
import os
import nltk

def process_files(quotes_file, entities_file, tokens_file):
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

    # Extracting text surrounding quotes
    def extract_surrounding_text(quote_start, quote_end, buffer=200):
        start_index = max(0, quote_start - buffer)
        end_index = quote_end + buffer
        surrounding_tokens = df_tokens[(df_tokens['token_ID_within_document'] >= start_index) & (df_tokens['token_ID_within_document'] <= end_index)]
        surrounding_text = ' '.join(surrounding_tokens['word'].tolist())
        return surrounding_text

    # Write the formatted data to quotes.csv, modified to include surrounding text
    output_filename = os.path.join(os.path.dirname(quotes_file), "quotes_modified.csv")
    with open(output_filename, 'w', newline='') as outfile:
        fieldnames = ["Text", "Start Location", "End Location", "Is Quote", "Speaker", "Text Quote Is Contained In"]
        writer = pd.DataFrame(columns=fieldnames)

        for index, row in df_quotes.iterrows():
            char_id = row['char_id']

            if not re.search('[a-zA-Z0-9]', row['quote']):
                print(f"Removing row with text: {row['quote']}")
                continue

            if character_info[char_id]["quote_count"] == 1:
                formatted_speaker = "Narrator"
            else:
                formatted_speaker = character_info[char_id]["formatted_speaker"] if char_id in character_info else "Unknown"
            
            surrounding_text = extract_surrounding_text(row['quote_start'], row['quote_end'])

            new_row = {"Text": row['quote'], "Start Location": row['quote_start'], "End Location": row['quote_end'], "Is Quote": "True", "Speaker": formatted_speaker, "Text Quote Is Contained In": surrounding_text}
            new_row_df = pd.DataFrame([new_row])
            writer = pd.concat([writer, new_row_df], ignore_index=True)

        writer.to_csv(output_filename, index=False)
        print(f"Saved quotes_modified.csv to {output_filename}")

def main():
    # Use glob to get all .quotes and .entities files within the "Working_files" directory and its subdirectories
    quotes_files = glob.glob('*.quotes', recursive=True)
    entities_files = glob.glob('*.entities', recursive=True)
    tokens_files = glob.glob('*.tokens', recursive=True)

    # Pair and process .quotes, .entities, and .tokens files with matching filenames (excluding the extension)
    for q_file in quotes_files:
        base_name = os.path.splitext(os.path.basename(q_file))[0]
        matching_entities_files = [e_file for e_file in entities_files if os.path.splitext(os.path.basename(e_file))[0] == base_name]
        matching_tokens_files = [t_file for t_file in tokens_files if os.path.splitext(os.path.basename(t_file))[0] == base_name]

        if matching_entities_files and matching_tokens_files:
            process_files(q_file, matching_entities_files[0], matching_tokens_files[0])

    print("All processing complete!")

if __name__ == "__main__":
    main()
