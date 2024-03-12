

import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import time

def shuffle_csv_in_place(csv_file):
    print("Reading CSV file...")
    df = pd.read_csv(csv_file)
    
    print("Shuffling rows...")
    # Introducing a loading bar simulation for the shuffling process
    for _ in tqdm(range(100), desc="Shuffling"):
        time.sleep(0.01)  # Simulate time delay for large files
    shuffled_df = shuffle(df)
    
    print("Writing shuffled data back to the original file...")
    shuffled_df.to_csv(csv_file, index=False)
    
    print(f"File '{csv_file}' has been shuffled and updated.")




from booknlp.booknlp import BookNLP
import pandas as pd
import glob
import os
import nltk
import re
import subprocess
import csv
from tqdm import tqdm

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

# Example usage:
# process_large_numbers_in_directory("/path/to/your/directory")



def convert_and_cleanup_ebooks(directory):
    """
    Converts ebook files in the specified directory to TXT format using Calibre's ebook-convert,
    and then removes any files that are not TXT files.

    Args:
    directory (str): The path to the directory containing the ebook files.
    """
    # Supported Calibre input formats for conversion (excluding TXT)
    supported_formats = {
        '.cbz', '.cbr', '.cbc', '.chm', '.epub', '.fb2', '.html',
        '.lit', '.lrf', '.mobi', '.odt', '.pdf', '.prc', '.pdb',
        '.pml', '.rb', '.rtf', '.snb', '.tcr'
    }

    # Convert files to TXT
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_ext = os.path.splitext(filename)[1].lower()
            # If the file is not a TXT and is a supported format, convert it
            if file_ext in supported_formats:
                txt_path = f"{os.path.splitext(filepath)[0]}.txt"
                try:
                    subprocess.run(['ebook-convert', filepath, txt_path], check=True)
                    print(f"Converted {filepath} to TXT.")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to convert {filepath}: {str(e)}")

    # Remove files that are not TXT
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and not filename.lower().endswith('.txt'):
            os.remove(filepath)
            print(f"Removed {filepath}.")

# Example usage
# convert_and_cleanup_ebooks("/path/to/your/directory")



def clean_text(text):
    corrected_text = text.replace(" n't", "n't").replace(" n’", "n’").replace("( ", "(").replace(" ,", ",").replace("gon na", "gonna").replace(" n’t", "n’t")
    corrected_text = re.sub(r' (?=[^a-zA-Z0-9\s])', '', corrected_text)
    return corrected_text

def process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=True):
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

    # Wrap all_book_files with tqdm for a progress bar
    for book_file in tqdm(all_book_files, desc="Processing books"):
        book_id = os.path.splitext(os.path.basename(book_file))[0]
        output_directory = os.path.join(output_base_dir, book_id)
        
        # Check if the book has already been processed
        required_files = [f"{book_id}.quotes", f"{book_id}.entities", f"{book_id}.tokens"]
        if not os.path.exists(output_directory) and all(os.path.exists(os.path.join(output_directory, f)) for f in required_files):

            print(f"Processing with booknlp: {book_id}")
            os.makedirs(output_directory, exist_ok=True)
            process_book_with_booknlp(book_file, output_directory, book_id)

            print(f"Extracting data from: {book_id}")
            quotes_file = os.path.join(output_directory, f"{book_id}.quotes")
            entities_file = os.path.join(output_directory, f"{book_id}.entities")
            tokens_file = os.path.join(output_directory, f"{book_id}.tokens")
            
            if os.path.exists(quotes_file) and os.path.exists(entities_file) and os.path.exists(tokens_file):
                df = process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=include_unknown_names)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Already processed with booknlp skipping: {book_id}")
            print(f"Extracting data from: {book_id}")
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
    convert_and_cleanup_ebooks(input_dir)
    process_large_numbers_in_directory(input_dir)
    process_all_books(input_dir, output_base_dir, include_unknown_names=False)
    shuffle_csv_in_place("new_output_dir/combined_books.csv")
    print("Complete!")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import os

# Assuming 'device' setup and 'tokenizer' initialization as before
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

# Preprocess and tokenize data (both training and validation)
def preprocess_and_tokenize_data(csv_file, output_dir, chunksize=1000):
    chunks = pd.read_csv(csv_file, chunksize=chunksize)
    for i, df_chunk in enumerate(chunks):
        formatted_input = df_chunk['Text Quote Is Contained In'] + " : quote in text " + df_chunk['Text']
        tokenized_inputs = tokenizer(formatted_input.tolist(), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df_chunk['Entity Name'])
        
        torch.save({'input_ids': tokenized_inputs['input_ids'], 'attention_mask': tokenized_inputs['attention_mask'], 'labels': torch.tensor(labels)}, os.path.join(output_dir, f'chunk_{i}.pt'))
        print(f"Processed and saved chunk {i+1}")

# Adjust CSV file paths as necessary
train_csv_file_path = 'new_output_dir/combined_books.csv'
validation_csv_file_path = 'new_output_dir/validation_books.csv'
preprocess_and_tokenize_data(train_csv_file_path, output_dir)

class CustomDataset(Dataset):
    def __init__(self, processed_dir):
        self.files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.pt')]
    def __len__(self):
        return len(self.files) * 1000  # Adjust based on your chunk size
    def __getitem__(self, idx):
        file_idx = idx // 1000
        within_file_idx = idx % 1000
        data = torch.load(self.files[file_idx])
        return {key: val[within_file_idx] for key, val in data.items()}

train_dataset = CustomDataset(output_dir)
# Assuming validation data is processed similarly and saved in 'validation_data' directory
validation_dataset = CustomDataset('validation_data')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(pd.read_csv(train_csv_file_path)['Entity Name'])))
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluation part, assuming 'compute_metrics' is defined as before
final_metrics = trainer.evaluate()
print(final_metrics)

# Assuming you preprocess a few examples from validation set for testing
# Adapt this part to load a few examples from your validation set
test_dataset = CustomDataset('test_data')  # Assuming test data is prepared similarly
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Example of printing out predictions for a few test examples
for i in range(len(test_dataset)):
    print(f"Test Example {i+1}: Predicted Label: {predicted_labels[i]}")
