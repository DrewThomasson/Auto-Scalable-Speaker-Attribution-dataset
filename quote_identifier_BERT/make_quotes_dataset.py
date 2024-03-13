from booknlp.booknlp import BookNLP
import pandas as pd
import glob
import os
import nltk
import re
import subprocess
import csv
from tqdm import tqdm





def clean_text(text):
    corrected_text = text.replace(" n't", "n't").replace(" n’", "n’").replace("( ", "(").replace(" ,", ",").replace("gon na", "gonna").replace(" n’t", "n’t").replace("“ ", "“")
    corrected_text = re.sub(r' (?=[^a-zA-Z0-9\s])', '', corrected_text)
    return corrected_text

def process_files(quotes_file, entities_file, tokens_file, output_file_path, IncludeUnknownNames=True):
    # Load the files
    #df_quotes = pd.read_csv(quotes_file, delimiter="\t")
    #df_entities = pd.read_csv(entities_file, delimiter="\t")
    #df_tokens = pd.read_csv(tokens_file, delimiter="\t")

    #This should fix any bad line errors by making it just skip any bad lines
    try:
        df_quotes = pd.read_csv(quotes_file, delimiter="\t", quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_entities = pd.read_csv(entities_file, delimiter="\t", quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_tokens = pd.read_csv(tokens_file, delimiter="\t", quoting=csv.QUOTE_NONE, error_bad_lines=False)
    except pd.errors.ParserError as e:
        print(f"Error reading one of the files: {e}")

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
    #for index, row in df_quotes.iterrows():
    for index, row in tqdm(df_quotes.iterrows(), total=df_quotes.shape[0], desc="Processing Quotes"):

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
    #for index, row in df_entities.iterrows():
    for index, row in tqdm(df_entities.iterrows(), total=df_entities.shape[0], desc="Processing Entities"):

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

    # Write the formatted data to quotes_dataset.csv, including text cleanup
    output_filename = os.path.join(os.path.dirname(quotes_file), "quotes_dataset.csv")
    with open(output_filename, 'w', newline='') as outfile:
        fieldnames = ["Text", "Start Location", "End Location", "Is Quote", "Speaker", "Context", "Entity Name"]
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
                "Context": surrounding_text,
                "Entity Name": entity_name
            }
            new_row_df = pd.DataFrame([new_row])
            writer = pd.concat([writer, new_row_df], ignore_index=True)

        #return writer  # This returns the DataFrame instead of saving it
        #print(f"Saved quotes_dataset.csv to {output_filename}")
        #output_filename = os.path.join(os.path.dirname(quotes_file), "quotes_dataset.csv")
        writer.to_csv(output_file_path, index=False)
        print(f"Saved quotes_dataset.csv to {output_file_path}")


"""
quotes_files = glob.glob('*.quotes', recursive=True)
entities_files = glob.glob('*.entities', recursive=True)
tokens_files = glob.glob('*.tokens', recursive=True)

# Make sure at least one file is found for each type, then select the first file
if quotes_files and entities_files and tokens_files:
    quotes_file = quotes_files[0]
    entities_file = entities_files[0]
    tokens_file = tokens_files[0]
    process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=True)
else:
    print("Error: Missing one or more required files.")


"""

"""
print("Removing any quotation marks from dataset...")
import pandas as pd
import re

# Load the CSV file
file_path = 'quotes_dataset.csv'  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Define a function to remove special characters
def remove_special_characters(text):
    # This regex matches any character that is NOT a letter, number, or space
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Apply the function to the "Context" and "Text" columns
df['Context'] = df['Context'].apply(remove_special_characters)
df['Text'] = df['Text'].apply(remove_special_characters)

# Save the cleaned data back to a CSV file
output_file_path = 'quotes_dataset.csv'  # Change this to your desired output file path
df.to_csv(output_file_path, index=False)

print("Special characters removed and data saved to", output_file_path)
"""
#for removing any quotations from the csv file

import pandas as pd
import re

def clean_csv(input_file_path, output_file_path):
    print("Removing any quotation marks from dataset...")
    
    # Load the CSV file
    df = pd.read_csv(input_file_path)

    # Define a function to remove special characters
    def remove_special_characters(text):
        # This regex matches any character that is NOT a letter, number, or space
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Apply the function to the "Context" and "Text" columns
    df['Context'] = df['Context'].apply(remove_special_characters)
    df['Text'] = df['Text'].apply(remove_special_characters)

    # Save the cleaned data back to a CSV file
    df.to_csv(output_file_path, index=False)

    print(f"Special characters removed and data saved to {output_file_path}")


"""

# Example usage:
input_file_path = 'quotes_dataset.csv'  # Change this to the path of your CSV file
output_file_path = 'cleaned_quotes_dataset.csv'  # Change this to your desired output file path
clean_csv(input_file_path, output_file_path)

"""


"""
print("Splitting sentences for column 'Text'...")
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Load the CSV file
df = pd.read_csv('quotes_dataset.csv')  # Change 'your_file.csv' to the path of your CSV file

# Function to split the text into sentences and duplicate the row for each sentence
def split_text_into_sentences(row):
    sentences = sent_tokenize(row['Text'])
    return [row.drop('Text').to_dict() | {'Text': sentence} for sentence in sentences]

# Apply the function and explode the DataFrame to separate rows
split_rows = df.apply(split_text_into_sentences, axis=1).explode().reset_index(drop=True)

# Convert the list of dictionaries back into a DataFrame
new_df = pd.DataFrame.from_records(split_rows)

# Save the expanded DataFrame to a new CSV file
new_df.to_csv('expanded_quotes_file.csv', index=False)  # Change 'expanded_file.csv' to your desired output file name

print("File has been expanded and saved.")


"""







import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt')

def split_sentences_and_expand_csv(input_file_path, output_file_path):
    print("Splitting sentences for column 'Text'...")
    
    # Load the CSV file
    df = pd.read_csv(input_file_path)

    # Function to split the text into sentences and duplicate the row for each sentence
    def split_text_into_sentences(row):
        sentences = sent_tokenize(row['Text'])
        return [row.drop('Text').to_dict() | {'Text': sentence} for sentence in sentences]

    # Apply the function and explode the DataFrame to separate rows
    split_rows = df.apply(split_text_into_sentences, axis=1).explode().reset_index(drop=True)

    # Convert the list of dictionaries back into a DataFrame
    new_df = pd.DataFrame.from_records(split_rows)

    # Save the expanded DataFrame to a new CSV file
    new_df.to_csv(output_file_path, index=False)

    print(f"File has been expanded and saved to {output_file_path}.")
"""
# Example usage:
input_file_path = 'quotes_dataset.csv'  # Change this to the path of your CSV file
output_file_path = 'expanded_quotes_file.csv'  # Change this to your desired output file name
split_sentences_and_expand_csv(input_file_path, output_file_path)
"""