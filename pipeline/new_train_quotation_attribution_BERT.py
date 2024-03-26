#convert all the books to txt
#Create the dataset from the folder of books using booknlp
#expand dataset by sentence splitting by pauses, 
#remove special characters




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
    corrected_text = text.replace(" n't", "n't").replace(" n’", "n’").replace("( ", "(").replace(" ,", ",").replace("gon na", "gonna").replace(" n’t", "n’t").replace("“ ", "“")
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

    def extract_surrounding_text(quote_start, quote_end, buffer=150):
        start_index = max(0, quote_start - buffer)
        end_index = quote_end + buffer
        surrounding_tokens = df_tokens[(df_tokens['token_ID_within_document'] >= start_index) & (df_tokens['token_ID_within_document'] <= end_index)]
        words_chunk = ' '.join([str(token_row['word']) for index, token_row in surrounding_tokens.iterrows()])
        return clean_text(words_chunk)

    # Write the formatted data to quotes_modified.csv, including text cleanup
    output_filename = os.path.join(os.path.dirname(quotes_file), "quotes_modified.csv")
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
        if os.path.exists(output_directory) and all(os.path.exists(os.path.join(output_directory, f)) for f in required_files):

            print(f"Already processed with booknlp skipping: {book_id}")
            print(f"Extracting data from: {book_id}")
            quotes_file = os.path.join(output_directory, f"{book_id}.quotes")
            entities_file = os.path.join(output_directory, f"{book_id}.entities")
            tokens_file = os.path.join(output_directory, f"{book_id}.tokens")
            
            if os.path.exists(quotes_file) and os.path.exists(entities_file) and os.path.exists(tokens_file):
                from make_quotes_dataset import process_files_dataframe
                #df = process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=include_unknown_names)
                df = process_files_dataframe(quotes_file, entities_file, tokens_file, IncludeUnknownNames=include_unknown_names)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Processing with booknlp: {book_id}")
            os.makedirs(output_directory, exist_ok=True)
            process_book_with_booknlp(book_file, output_directory, book_id)

            print(f"Extracting data from: {book_id}")
            quotes_file = os.path.join(output_directory, f"{book_id}.quotes")
            entities_file = os.path.join(output_directory, f"{book_id}.entities")
            tokens_file = os.path.join(output_directory, f"{book_id}.tokens")
            
            if os.path.exists(quotes_file) and os.path.exists(entities_file) and os.path.exists(tokens_file):
                from make_quotes_dataset import process_files_dataframe
                df = process_files_dataframe(quotes_file, entities_file, tokens_file, IncludeUnknownNames=include_unknown_names)
                #df = process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=include_unknown_names)
                combined_df = pd.concat([combined_df, df], ignore_index=True)


    combined_csv_path = os.path.join(output_base_dir, "combined_books.csv")
    if not combined_df.empty:
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved to {combined_csv_path}")

def main():
    input_dir = 'books_small'
    output_base_dir = 'new_output_dir'
    convert_and_cleanup_ebooks(input_dir)
    process_large_numbers_in_directory(input_dir)
    process_all_books(input_dir, output_base_dir, include_unknown_names=False)
    #shuffle_csv_in_place("new_output_dir/combined_books.csv")
    from BERT_functions import replace_titles_and_abbreviations, revert_titles_and_abbreviations, split_csv_by_pauses, clean_csv_special_characters
    from make_quotes_dataset import split_csv_by_pauses
    split_csv_by_pauses("new_output_dir/combined_books.csv","new_output_dir/combined_books.csv")
    shuffle_csv_in_place("new_output_dir/combined_books.csv")
    #clean_csv_special_characters()


    print("Complete!")

if __name__ == "__main__":
    main()



































#condense dataset into 5 min repeats and 47 max quotes from each character (dont want the model to be cheating by memorizing names afterall)
print("condensing dataset...")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the CSV file
df = pd.read_csv('new_output_dir/combined_books.csv')  # Make sure to replace 'combined_books.csv' with the path to your file

# Step 2: Filter based on the "Entity Name" column
entity_counts = df['Entity Name'].value_counts()
filtered_entities = entity_counts[(entity_counts > 5) & (entity_counts < 47)].index
filtered_df = df[df['Entity Name'].isin(filtered_entities)]

# Step 3: Write the filtered DataFrame to a new CSV file
filtered_df.to_csv('simple_averaged_dataset.csv', index=False)

# Step 4: Generate Visualizations and Save as Images
plt.figure(figsize=(12, 6))

# Original Data Histogram
sns.histplot(entity_counts, binwidth=1, kde=True, label='Original', color='blue')
plt.title('Histogram of Entity Occurrences: Original')
plt.xlabel('Number of Occurrences')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('histogram_original.png')  # Save the figure
plt.close()  # Close the figure to free up memory

# Filtered Data Histogram
filtered_entity_counts = filtered_df['Entity Name'].value_counts()
if not filtered_entity_counts.empty:
    plt.figure(figsize=(12, 6))
    sns.histplot(filtered_entity_counts, binwidth=1, kde=True, color='red', label='Filtered')
    plt.title('Histogram of Entity Occurrences: Filtered')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('histogram_filtered.png')  # Save the figure
    plt.close()  # Close the figure
else:
    print("No entities in the filtered dataset meet the histogram plotting criteria.")

# Box Plot Comparing the Number of Entities per Row Before and After Filtering
plt.figure(figsize=(10, 5))
# Prepare data
original_counts = df['Entity Name'].map(df['Entity Name'].value_counts())
filtered_counts = filtered_df['Entity Name'].map(filtered_df['Entity Name'].value_counts())
sns.boxplot(data=[original_counts, filtered_counts], palette=['blue', 'red'])
plt.xticks([0, 1], ['Original', 'Filtered'])
plt.title('Box Plot of Entity Counts per Entity')
plt.ylabel('Number of Entities')
plt.savefig('boxplot_entity_counts.png')  # Save the figure
plt.close()  # Close the figure

print("dataset condensed continuing...")


df = filtered_df.copy()

























"""
#this is the code to train BERT on the new dataset
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('simple_averaged_dataset.csv')

# Adjusting column names in the DataFrame to include formatted inputs for BERT
df['formatted_input'] = df['Context'] + " : quote in text " + df['Text']

# Encoding entity names for classification
label_encoder = LabelEncoder()
df['entity_name_encoded'] = label_encoder.fit_transform(df['Entity Name'])

# Splitting the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)

# Initializing the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizing function that processes the data for BERT
def tokenize_function(examples):
    text_list = examples['formatted_input'].tolist()
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512)

# Tokenize the training and validation data
train_encodings = tokenize_function(train_df)
val_encodings = tokenize_function(val_df)

# Dataset class to hold the tokenized inputs and labels
class QuotationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets for training and validation
train_dataset = QuotationDataset(train_encodings, train_df['entity_name_encoded'].values)
val_dataset = QuotationDataset(val_encodings, val_df['entity_name_encoded'].values)

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)

# Define the compute_metrics function for evaluating the model
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Training arguments to customize the training process
training_args = TrainingArguments(
    output_dir='./results',  # Where to store the training outputs.
    num_train_epochs=3,  # Total number of training epochs.
    per_device_train_batch_size=8,  # Batch size for training.
    per_device_eval_batch_size=16,  # Batch size for evaluation.
    evaluation_strategy="steps",  # Evaluate every `eval_steps`.
    eval_steps=200,  # Number of steps to run evaluation.
    save_strategy="steps",  # Align saving with evaluation strategy.
    save_steps=200,  # Save the model every 100 steps, align with eval for convenience.
    fp16=torch.cuda.is_available(),  # Use mixed precision if available.
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler.
    weight_decay=0.01,  # Weight decay for optimization.
    logging_dir='./logs',  # Directory for storing logs.
    logging_steps=10,  # Log metrics every 10 steps.
    report_to="none"  # Disable reporting to external services.
)

# Initialize the Trainer with the model, training arguments, datasets, and compute_metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
final_metrics = trainer.evaluate()
print(final_metrics)

# Use the formatted input for test prediction examples
test_cases_formatted = val_df['formatted_input'].tolist()[:3]
test_encodings = tokenize_function(pd.DataFrame({'formatted_input': test_cases_formatted}))
test_dataset = QuotationDataset(test_encodings, [0] * len(test_cases_formatted))

# Perform predictions on the test dataset
with torch.no_grad():
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

# Print out the formatted input and the predicted entity for each test case
for formatted_input, label in zip(test_cases_formatted, predicted_labels):
    print(f"Formatted Input: {formatted_input}")
    print(f"Predicted Entity: {label_encoder.inverse_transform([label])[0]}")
"""


"""
#training with GPT-2
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv('simple_averaged_dataset.csv')
texts = df['Context'] + " : quote in text " + df['Text']  # Concatenate context and text
texts.to_csv('formatted_texts.txt', header=False, index=False)  # Save concatenated texts to a file

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Prepare dataset and dataloader
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./formatted_texts.txt",
    block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")
"""


#infrence code for GPT-2
"""
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model_path = "./gpt2_finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Generate text
prompt = "Here's a scenario to consider: "  # Example prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

# Generate output
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=100,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    do_sample=True,
    num_return_sequences=1,
)

# Decode the output
generated_sequence = tokenizer.decode(output_sequences[0], clean_up_tokenization_spaces=True)
print(generated_sequence)

"""
"""
#training code for Distil GPT-2

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# Load your dataset
df = pd.read_csv('new_output_dir/combined_books.csv')  # Make sure to have your dataset at this path
df['text'] = df['Context'] + " " + df['Text']  # Assuming you have 'Context' and 'Text' columns

# Split the dataset into training and validation
train_df, val_df = train_test_split(df, test_size=0.1)

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilgpt2')

# Tokenize the text
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True)

# Create a custom dataset
class GPTDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = GPTDataset(train_encodings)
val_dataset = GPTDataset(val_encodings)

# Load DistilGPT-2 model
model = DistilBertForSequenceClassification.from_pretrained('distilgpt2', num_labels=1)  # Adjust num_labels as per your requirement

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Adjust based on your VRAM and model size
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

"""






#infrence code for distil gpt-2
"""
from transformers import DistilGPT2LMHeadModel, GPT2Tokenizer

# Load trained model and tokenizer
model = DistilGPT2LMHeadModel.from_pretrained('./results')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

model.eval()

# Generate text
prompt = "The future of AI in healthcare"
inputs = tokenizer.encode(prompt, return_tensors='pt')

# Generate text using the model
output = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode generated sequence
text = tokenizer.decode(output[0], skip_special_tokens=True)
print(text)






"""

#FUCK SHIT DREW LOOK HERE
#ask GPT-4 for this

"""
I want to give distil-GPT-2 a dataframe and have it use the features "Context" and "Text" to trey to predict the next word value, and the corretc answer for each row is held in the column named "Speaker"
the correct answer is a speaker name of course this is not a classificiation thing its predict next workd owr sentence whatnot

the point is to fine tune the model to get goo at specioially the task of predictiing the next speaker

format will be 

input:[''Context' :Quote in text: ''Text']

output:['Speaker']

fromt eh dataframe

"""

























'''
#training code for Distil GPT-2 
#chat it came from https://chat.openai.com/share/2a846c4a-0dca-46a6-9de9-8dbd629d39cd
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import load_metric
import numpy as np

# Assuming df is your DataFrame
# Make sure to load your DataFrame before this step
df['formatted_input'] = df['Context'] + " ::: " + df['Text']
df = df[['formatted_input', 'Entity Name']]

#save the df as csv file for testing idk
df.to_csv('GPT-2_training_df.csv', index=False)

# Load tokenizer and set pad token
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set tokenizer's pad token

# Encode the labels
label_encoder = LabelEncoder()
df['encoded_speaker'] = label_encoder.fit_transform(df['Entity Name'])

# Split the data
train_df, eval_df = train_test_split(df, test_size=0.1)

# Tokenize the inputs
def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=512)

train_dataset = tokenize_function(train_df['formatted_input'].tolist())
eval_dataset = tokenize_function(eval_df['formatted_input'].tolist())

train_labels = train_df['encoded_speaker'].tolist()
eval_labels = eval_df['encoded_speaker'].tolist()

# Convert to PyTorch dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_dataset, train_labels)
eval_dataset = CustomDataset(eval_dataset, eval_labels)

# Load model and set pad token id
model = GPT2ForSequenceClassification.from_pretrained('distilgpt2', num_labels=len(label_encoder.classes_))
model.config.pad_token_id = tokenizer.pad_token_id  # Make sure model recognizes pad_token_id

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics  # Pass the compute_metrics function
)

# Train the model
trainer.train()


#this should test the trained model at the end
#hopefully this should work as infrence code lol

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Load the trained model from the checkpoint
model_path = "results/checkpoint-3500"
model = GPT2ForSequenceClassification.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Select 5 rows from the DataFrame
sample_df = df.sample(n=5)

# Tokenize the inputs
sample_inputs = tokenize_function(sample_df['formatted_input'].tolist())

# Create a CustomDataset for the sample inputs
sample_dataset = CustomDataset(sample_inputs, sample_df['encoded_speaker'].tolist())

# Set the model to evaluation mode
model.eval()

# Iterate over the sample dataset
for i in range(len(sample_dataset)):
    item = sample_dataset[i]
    input_ids = item['input_ids'].unsqueeze(0).to(model.device)
    attention_mask = item['attention_mask'].unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()

    # Convert encoded labels back to original labels
    predicted_speaker = label_encoder.inverse_transform([predicted_label])[0]
    correct_speaker = label_encoder.inverse_transform([item['labels'].item()])[0]

    print(f"Input {i+1}:")
    print(sample_df['formatted_input'].iloc[i])
    print(f"Predicted Speaker: {predicted_speaker}")
    print(f"Correct Speaker: {correct_speaker}")
    print("---")

    






'''



#this is the more even new way of training the distil-GPT-2 model on the books
import pandas as pd
from datasets import load_metric
import numpy as np
# Assuming df is your DataFrame
# Make sure to load your DataFrame before this step
df['formatted_input'] = df['Context'] + " ::: " + df['Text'] + " <speaker>"
df['Entity Name'] = df['Entity Name'] + "<end>"
df = df[['formatted_input', 'Entity Name']]

#save the df as csv file for testing idk
df.to_csv('GPT-2_training_df.csv', index=False)






#new training code for Distil-GPT-2


import pandas as pd
from datasets import Dataset
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the CSV file into a DataFrame
df = pd.read_csv("GPT-2_training_df.csv")

# Create a Hugging Face Dataset from the DataFrame
dataset = Dataset.from_pandas(df)

# Filter out rows where either 'Input_string' or 'Output_string' is missing
dataset = dataset.filter(lambda example: example['formatted_input'] is not None and example['Entity Name'] is not None)

# Function to concatenate the input and output strings
def concatenate_examples(examples):
    return {'text': examples['formatted_input'] + " " + examples['Entity Name']}

# Apply the function to the dataset
dataset = dataset.map(concatenate_examples)

# Load the pre-trained DistilGPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Explicitly set the tokenizer's pad token to the EOS token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=600)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set the format to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=1,
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer to the "trained_model" folder
model_save_path = "./trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)












#this is the infrence code for the trained gpt-2 model

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

# Load the trained model

model_path = "./trained_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Function to generate the model's output given an input string
def generate_output(input_string):
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True, max_length=1000)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output = model.generate(
        input_ids,
        max_length=1000,  # Further reduce max length if expecting shorter responses
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.5,  # Adjust temperature for determinism
        top_k=50,  # Tighten top_k
        top_p=0.95,  # Slightly loosen top_p for a bit of variability
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    generated_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_output



import re

def extract_first_value(text, start_delim="<speaker>", end_delim="<end>"):
    # Create a regular expression pattern to find text between the delimiters
    pattern = re.escape(start_delim) + "(.*?)" + re.escape(end_delim)
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the first group (the text between the delimiters)
    if match:
        return match.group(1)  # `group(1)` refers to the text captured by `(.*?)`
    else:
        return "No match found"


num_of_tests = 5
correct_responces = 0
for i in range(num_of_tests):
    # get random row values from dataframe 
    formatted_input, formatted_output = pd.read_csv("GPT-2_training_df.csv").sample(1)[["formatted_input", 'Entity Name']].values[0] 
    #print("INPUT")
    #print(formatted_input)
    #print("OUTPUT")
    #print(formatted_output)

    # Generate the model's output
    output_string = generate_output(formatted_input)

    # Print the generated output
    print("Generated Output:")
    #print(output_string)
    #print("Extracting first answer from output...")
    #print("Generated answer")
    extracted_gen_answer = extract_first_value(output_string)
    print(extracted_gen_answer)


    #print the correct output
    print("Correct output:")
    formatted_output = formatted_output.replace('<end>', '')
    print(formatted_output)


    #compare the correct output with the generated output
    if extracted_gen_answer.replace(" ", "") == formatted_output.replace(" ", ""):
        correct_responces = correct_responces +1
        print("Correct!")
print(f'Accuracy score of : {correct_responces/num_of_tests}')
'''
# Get the input string from the user
input_string = input("Enter an input string: ")

# Generate the model's output
output_string = generate_output(input_string)

# Print the generated output
print("Generated Output:")
print(output_string)

'''





