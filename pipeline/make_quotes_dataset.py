from booknlp.booknlp import BookNLP
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
    """
    # Function to split text based on pauses
    def split_text(text):
        # Define the pattern for splitting: period, comma, exclamation, semicolon
        pattern = r'[.!,;?:]'
        # Split text and filter out any empty strings from the list
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        # Add the punctuation back to the parts except the last one
        parts = [part + text[text.find(part) + len(part)] if text.find(part) + len(part) < len(text) and text[text.find(part) + len(part)] in '.!,;?' else part for part in parts]
        return parts"""

    def split_text(text):
        # Define the pattern for splitting: period, comma, exclamation, semicolon
        # Modified to handle consecutive punctuation as a single split point
        pattern = r'(?<!\s)[.!,;?:"]+(?=\s|$)'
        
        # Split text and filter out any empty strings from the list
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        
        # Determine the punctuation to add back based on the original text
        def punctuation_to_add(index, part):
            punctuation = ''
            if text.find(part) + len(part) < len(text):
                # Look ahead for punctuation not followed by a space (but not at the end of text)
                look_ahead_index = text.find(part) + len(part)
                while look_ahead_index < len(text) and text[look_ahead_index] in '.!,;?"' and text[look_ahead_index] not in ' ':
                    punctuation += text[look_ahead_index]
                    look_ahead_index += 1
            return punctuation
        
        # Add the punctuation back to the parts except the last one
        parts_with_punctuation = [part + punctuation_to_add(i, part) for i, part in enumerate(parts)]
        
        return parts_with_punctuation



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

def process_files(quotes_file, entities_file, tokens_file, output_file_path, IncludeUnknownNames=True):
    # Load the files
    #df_quotes = pd.read_csv(quotes_file, delimiter="\t")
    #df_entities = pd.read_csv(entities_file, delimiter="\t")
    #df_tokens = pd.read_csv(tokens_file, delimiter="\t")

    #This should fix any bad line errors by making it just skip any bad lines
    try:
        df_quotes = pd.read_csv(quotes_file, delimiter="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        df_entities = pd.read_csv(entities_file, delimiter="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        df_tokens = pd.read_csv(tokens_file, delimiter="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip')
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

    def extract_surrounding_text(quote_start, quote_end, buffer=150):
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



#makes quotes file but returns a dataframe instead of making a csv

def process_files_dataframe(quotes_file, entities_file, tokens_file, IncludeUnknownNames=True):
    # Load the files
    #df_quotes = pd.read_csv(quotes_file, delimiter="\t")
    #df_entities = pd.read_csv(entities_file, delimiter="\t")
    #df_tokens = pd.read_csv(tokens_file, delimiter="\t")

    #This should fix any bad line errors by making it just skip any bad lines
    try:
        df_quotes = pd.read_csv(quotes_file, delimiter="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        df_entities = pd.read_csv(entities_file, delimiter="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        df_tokens = pd.read_csv(tokens_file, delimiter="\t", quoting=csv.QUOTE_NONE, on_bad_lines='skip')
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

    def extract_surrounding_text(quote_start, quote_end, buffer=150):
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

        return writer  # This returns the DataFrame instead of saving it
        print(f"Saved quotes_dataset.csv to {output_filename}")
        output_filename = os.path.join(os.path.dirname(quotes_file), "quotes_dataset.csv")
        #writer.to_csv(output_file_path, index=False)
        #print(f"Saved quotes_dataset.csv to {output_file_path}")

        #new_row_df = pd.DataFrame([new_row])
        #writer = pd.concat([writer, new_row_df], ignore_index=True)

        #return writer  # This returns the DataFrame instead of saving it
        #print("Returned quotes dataframe")




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




from booknlp.booknlp import BookNLP
import pandas as pd
import glob
import os
import nltk
import re
import subprocess
import csv
from tqdm import tqdm



def process_large_numbers_in_txt(file_path):
    """
    Removes commas from large numbers in the specified TXT file.
    Args:
    file_path (str): The path to the TXT file to process.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match numbers with commas
    pattern = r'\b\d{1,3}(,\d{3})+\b'

    # Remove commas in numerical sequences
    modified_content = re.sub(pattern, lambda m: m.group().replace(',', ''), content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)








print("processing books folder...")
def process_large_numbers_in_directory(directory):
    """
    Processes all TXT files in the given directory, removing commas from large numbers.

    Args:
    directory (str): The path to the directory containing the TXT files.
    """

    def process_large_numbers_in_txt(file_path):
        """
        Removes commas from large numbers in the specified TXT file.

        Args:
        file_path (str): The path to the TXT file to process.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regular expression to match numbers with commas
        pattern = r'\b\d{1,3}(,\d{3})+\b'

        # Remove commas in numerical sequences
        modified_content = re.sub(pattern, lambda m: m.group().replace(',', ''), content)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory, filename)
            process_large_numbers_in_txt(file_path)
            print(f"Processed large numbers in: {filename}")
