
'''

#I want a program that will take all the books i have in txt format in a folder called "books/"
#and then use booknlp to extract all of the metatdata files from those books, such as the .tokens, .quotes, and .entites files
#Then use the methods and functions shown below to creata a quotes.csv and a non_quotes.csv file for each of the books that have been processed by booknlp
#Then combine all of the created quotes and non_quotes csv files for every book into a giant csv file




#example of how to use booknlp to get the .tokens, .entites, and .quotes files from any book in a txt format
"""
from booknlp.booknlp import BookNLP

model_params={
		"pipeline":"entity,quote,supersense,event,coref", 
		"model":"big"
	}
	
booknlp=BookNLP("en", model_params)

# Input file to process
input_file="input_dir/bartleby.txt"

# Output directory to store resulting files in
output_directory="output_dir/bartleby/" #this will hold the .quotes, .tokens, and .entites files for that paticular book, so its where they will be stored

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id="bartleby"

booknlp.process(input_file, output_directory, book_id)
"""



#To create the non_quotes.csv file
from make_non_quotes_dataset import clean_text
from make_non_quotes_dataset import split_into_sentences
from make_non_quotes_dataset import process_files
from make_non_quotes_dataset import clean_csv_special_characters
import pandas as pd
import re
import glob
import os
from tqdm import tqdm  # Import tqdm



# How to use those functions 

quotes_files = glob.glob('*.quotes', recursive=True)
tokens_files = glob.glob('*.tokens', recursive=True)
output_filePath = 'non_quotes.csv'

print("Starting processing...")
for q_file in tqdm(quotes_files, desc="Overall Progress"):
    base_name = os.path.splitext(os.path.basename(q_file))[0]
    matching_token_files = [t_file for t_file in tokens_files if os.path.splitext(os.path.basename(t_file))[0] == base_name]

    if matching_token_files:
        process_files(q_file, matching_token_files[0], output_filePath)

print("All processing complete!")




# Example usage:
input_file_path = 'non_quotes.csv'  # Replace with your input file path
output_file_path = 'non_quotes.csv'  # Replace with your desired output file path
columns_to_clean = ['Context', 'Text']  # Columns to clean

clean_csv_special_characters(input_file_path, output_file_path, columns_to_clean)
#and now the non_quotes.csv file is created cleaned up and is ready!









#This code will create the quotes.csv file
from booknlp.booknlp import BookNLP
import pandas as pd
import glob
import os
import nltk
import re
import subprocess
import csv
from tqdm import tqdm


from make_quotes_dataset import clean_text
from make_quotes_dataset import process_files

#Example of how to use 
quotes_files = glob.glob('*.quotes', recursive=True)
entities_files = glob.glob('*.entities', recursive=True)
tokens_files = glob.glob('*.tokens', recursive=True)
output_file_path = "quotes_dataset.csv"

# Make sure at least one file is found for each type, then select the first file
if quotes_files and entities_files and tokens_files:
    quotes_file = quotes_files[0]
    entities_file = entities_files[0]
    tokens_file = tokens_files[0]
    process_files(quotes_file, entities_file, tokens_file, IncludeUnknownNames=True, output_file_path=output_file_path)
    #process_files(quotes_file, entities_file, tokens_file, output_file_path, IncludeUnknownNames=True)

else:
    print("Error: Missing one or more required files.")

print("Removing any quotation marks from dataset...")
import pandas as pd
import re

from make_quotes_dataset import clean_csv


# Example usage:
input_file_path = 'quotes_dataset.csv'  # Change this to the path of your CSV file
output_file_path = 'quotes_dataset.csv'  # Change this to your desired output file path
clean_csv(input_file_path, output_file_path)


import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt')


from make_quotes_dataset import split_sentences_and_expand_csv

# Example usage:
input_file_path = 'quotes_dataset.csv'  # Change this to the path of your CSV file
output_file_path = 'quotes_dataset.csv'  # Change this to your desired output file name
split_sentences_and_expand_csv(input_file_path, output_file_path)
#and now the quotes_dataset.csv file had been created and cleaned up and is ready!







'''




from booknlp.booknlp import BookNLP
import pandas as pd
import glob
import os
from tqdm import tqdm

# Import your dataset creation scripts
from make_non_quotes_dataset import clean_text, split_into_sentences, process_files as process_non_quotes_files, clean_csv_special_characters
from make_quotes_dataset import process_files as process_quotes_files, clean_csv, split_sentences_and_expand_csv

# Download necessary NLTK data
import nltk
nltk.download('punkt')

def process_book(input_file, output_directory, book_id):
    model_params={
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "big"
    }

    booknlp = BookNLP("en", model_params)
    booknlp.process(input_file, output_directory, book_id)

def process_all_books(input_dir='books/'):
    for input_file in glob.glob(os.path.join(input_dir, '*.txt')):
        book_name = os.path.splitext(os.path.basename(input_file))[0]
        output_directory = f'output_dir/{book_name}/'
        os.makedirs(output_directory, exist_ok=True)
        
        # Process the book with BookNLP to generate .quotes, .tokens, and .entities files
        process_book(input_file, output_directory, book_name)
        
        # Find the generated files
        quotes_file_path = glob.glob(os.path.join(output_directory, f'{book_name}.quotes'))[0]
        tokens_file_path = glob.glob(os.path.join(output_directory, f'{book_name}.tokens'))[0]
        entities_file_path = glob.glob(os.path.join(output_directory, f'{book_name}.entities'))[0]
        
        # Assuming your process_files functions are correctly defined and work with the paths
        # Generate non_quotes.csv for the current book
        non_quotes_output_filePath = os.path.join(output_directory, f'{book_name}_non_quotes.csv')
        ##process_non_quotes_files(quotes_file_path, tokens_file_path, non_quotes_output_filePath)  # Pass correct file paths
        from make_non_quotes_dataset import clean_text
        from make_non_quotes_dataset import split_into_sentences
        from make_non_quotes_dataset import process_files
        from make_non_quotes_dataset import clean_csv_special_characters
        process_files(quotes_file_path, tokens_file_path, non_quotes_output_filePath)
        columns_to_clean = ['Context', 'Text']  # Columns to clean
        clean_csv_special_characters(non_quotes_output_filePath, non_quotes_output_filePath, columns_to_clean)

        
        # Generate quotes.csv for the current book
        quotes_output_filePath = os.path.join(output_directory, f'{book_name}_quotes.csv')
        ##process_quotes_files(quotes_file_path, entities_file_path, tokens_file_path, quotes_output_filePath)  # Pass correct file paths
        from make_quotes_dataset import clean_text
        from make_quotes_dataset import process_files
        output_file_path = quotes_output_filePath
        process_files(quotes_file_path, entities_file_path, tokens_file_path, IncludeUnknownNames=True, output_file_path=output_file_path)
        from make_quotes_dataset import clean_csv
        clean_csv(quotes_output_filePath, quotes_output_filePath)
        from make_quotes_dataset import split_sentences_and_expand_csv
        split_sentences_and_expand_csv(quotes_output_filePath, quotes_output_filePath)



"""
def combine_csvs(output_dir='output_dir/', combined_filename='combined_data.csv'):
    combined_non_quotes_df = pd.DataFrame()
    combined_quotes_df = pd.DataFrame()

    for non_quotes_file in glob.glob(os.path.join(output_dir, '*_non_quotes.csv')):
        df = pd.read_csv(non_quotes_file)
        combined_non_quotes_df = pd.concat([combined_non_quotes_df, df])

    for quotes_file in glob.glob(os.path.join(output_dir, '*_quotes.csv')):
        df = pd.read_csv(quotes_file)
        combined_quotes_df = pd.concat([combined_quotes_df, df])

    combined_non_quotes_df.to_csv(os.path.join(output_dir, 'combined_non_quotes.csv'), index=False)
    combined_quotes_df.to_csv(os.path.join(output_dir, 'combined_quotes.csv'), index=False)

if __name__ == "__main__":
    process_all_books()  # Process all books to extract metadata and create individual CSVs
    combine_csvs()  # Combine all individual CSVs into two giant CSVs
"""
process_all_books()


import pandas as pd
import glob
import os

import pandas as pd
import os

def combine_csv_files(root_dir):
    all_csv_files = []

    # Walk through all directories and files starting from root_dir
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file is a CSV
            if file.endswith('.csv'):
                all_csv_files.append(os.path.join(subdir, file))

    # Define the required columns
    required_columns = ["Text", "Start Location", "End Location", "Is Quote", "Speaker", "Context"]
    optional_columns = ["Entity Name"]

    # Initialize an empty DataFrame to hold combined data
    combined_df = pd.DataFrame()

    # Process each CSV file
    for csv_file in all_csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Ensure all required columns are present, fill missing ones with NA
            for col in required_columns:
                if col not in df.columns:
                    df[col] = "NA"

            # Check and add the optional column if missing
            for col in optional_columns:
                if col not in df.columns:
                    df[col] = "Probs Narrator"

            # Append to the combined DataFrame
            combined_df = pd.concat([combined_df, df[required_columns + optional_columns]], ignore_index=True)
        
        except pd.errors.EmptyDataError:
            print(f"Skipping {csv_file} due to being empty or improperly formatted.")

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv('combined_csv.csv', index=False)

    return combined_df

# Example usage
# Replace 'path_to_your_directory' with the actual path to the directory containing your CSV files
root_dir = "output_dir"
combined_data = combine_csv_files(root_dir)





#this code will shuffle the output csv file
import pandas as pd
from sklearn.utils import shuffle

def shuffle_csv_rows(input_csv_path, output_csv_path=None):
    """
    Shuffle the rows of a CSV file.

    Parameters:
    - input_csv_path: str, the path to the input CSV file.
    - output_csv_path: str, optional, the path to save the shuffled CSV file.
                       If not provided, the input CSV file will be overwritten.

    Returns:
    None
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)
    
    # Shuffle the DataFrame rows
    shuffled_df = shuffle(df)
    
    # Reset index to avoid saving the old index as a column in the new CSV
    shuffled_df.reset_index(drop=True, inplace=True)
    
    # Determine the output path
    if output_csv_path is None:
        output_csv_path = input_csv_path
    
    # Save the shuffled DataFrame to a CSV file
    shuffled_df.to_csv(output_csv_path, index=False)

# Example usage:
shuffle_csv_rows('combined_csv.csv', 'combined_csv.csv')











import pandas as pd
from sklearn.utils import shuffle

def even_out_and_shuffle_csv(input_csv_path, output_csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)
    
    # Filter rows where "Is Quote" is True or False
    true_df = df[df['Is Quote'] == True]
    false_df = df[df['Is Quote'] == False]
    
    # Find the minimum count between True and False rows
    min_count = min(len(true_df), len(false_df))
    
    # Sample min_count rows from each DataFrame to ensure they are even
    true_df_sampled = true_df.sample(n=min_count, random_state=1)
    false_df_sampled = false_df.sample(n=min_count, random_state=1)
    
    # Combine the sampled DataFrames
    combined_df = pd.concat([true_df_sampled, false_df_sampled])
    
    # Print out the count of "True" and "False" rows after balancing
    print(f'Number of "True" rows: {len(true_df_sampled)}')
    print(f'Number of "False" rows: {len(false_df_sampled)}')
    
    # Shuffle the combined DataFrame
    shuffled_df = shuffle(combined_df, random_state=1)
    
    # Save the modified DataFrame to a new CSV file
    shuffled_df.to_csv(output_csv_path, index=False)

# Example usage
input_csv = 'combined_csv.csv'  # Replace with the path to your input CSV
output_csv = 'combined_csv.csv'  # Replace with the path to your output CSV
even_out_and_shuffle_csv(input_csv, output_csv)
























'''



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

df = pd.read_csv('combined_csv.csv')

# Adjusting column names in the DataFrame to include formatted inputs for BERT
df['formatted_input'] = df['Context'] + " : Is Sentence Quote :" + df['Text']

# Encoding entity names for classification
label_encoder = LabelEncoder()
#df['entity_name_encoded'] = label_encoder.fit_transform(df['Is Quote'])
df['is_quote_encoded'] = label_encoder.fit_transform(df['Is Quote'])

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
train_dataset = QuotationDataset(train_encodings, train_df['is_quote_encoded'].values)
val_dataset = QuotationDataset(val_encodings, val_df['is_quote_encoded'].values)

## Load the BERT model for sequence classification
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_)).to(device)

# Load the BERT model configured for binary classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)


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
    eval_steps=500,  # Number of steps to run evaluation.
    save_strategy="steps",  # Align saving with evaluation strategy.
    save_steps=5000,  # Save the model every 100 steps, align with eval for convenience.
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





'''






































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

df = pd.read_csv('combined_csv.csv')

# Ensure 'Context' and 'Text' are strings
df['Context'] = df['Context'].astype(str)
df['Text'] = df['Text'].astype(str)

# You might need to handle NaN values here if they exist
# For example: df.fillna('Missing', inplace=True)

df['formatted_input'] = df['Context'] + " : Is Sentence Quote :" + df['Text']

# Encoding labels for classification
label_encoder = LabelEncoder()
df['is_quote_encoded'] = label_encoder.fit_transform(df['Is Quote'])

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
train_dataset = QuotationDataset(train_encodings, train_df['is_quote_encoded'].values)
val_dataset = QuotationDataset(val_encodings, val_df['is_quote_encoded'].values)

# Load the BERT model for sequence classification configured for binary classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Define the compute_metrics function for evaluating the model
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Training arguments to customize the training process
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1750,
    fp16=torch.cuda.is_available(),
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none"
)

# Initialize the Trainer
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

# Print out the formatted input and the predicted label for each test case
for formatted_input, label in zip(test_cases_formatted, predicted_labels):
    print(f"Formatted Input: {formatted_input}")
    print(f"Predicted Entity: {label_encoder.inverse_transform([label])[0]}")

























