


#I want a program that will take all the books i have in txt format in a folder called "books/"
#and then use booknlp to extract all of the metatdata files from those books, such as the .tokens, .quotes, and .entites files
#Then use the methods and functions shown below to creata a quotes.csv and a non_quotes.csv file for each of the books that have been processed by booknlp
#Then combine all of the created quotes and non_quotes csv files for every book into a giant csv file






import os

#This will convert every book in the books folder into a txt file

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



convert_and_cleanup_ebooks("books")


#FUCKKKKKK LOOK AT ME DREW POOP
#This will remove any commas from the txt files when it comes to commas in large didgets
from make_quotes_dataset import process_large_numbers_in_directory
process_large_numbers_in_directory("books")





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
        
        # Check if all required files already exist
        required_files = [f'{book_name}.quotes', f'{book_name}.tokens', f'{book_name}.entities']
        all_files_exist = all(os.path.exists(os.path.join(output_directory, file)) for file in required_files)
        
        if all_files_exist:
            print(f"Skipping {book_name} as it has already been processed.")
            continue
        
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
        #from make_quotes_dataset import clean_csv_special_characters
        from make_quotes_dataset import split_csv_by_pauses
        split_csv_by_pauses(non_quotes_output_filePath, non_quotes_output_filePath)
        columns_to_clean = ['Context', 'Text']  # Columns to clean
        clean_csv_special_characters(non_quotes_output_filePath, non_quotes_output_filePath, columns_to_clean)
        
        # Generate quotes.csv for the current book
        quotes_output_filePath = os.path.join(output_directory, f'{book_name}_quotes.csv')
        ##process_quotes_files(quotes_file_path, entities_file_path, tokens_file_path, quotes_output_filePath)  # Pass correct file paths
        from make_quotes_dataset import clean_text
        from make_quotes_dataset import process_files
        output_file_path = quotes_output_filePath
        process_files(quotes_file_path, entities_file_path, tokens_file_path, IncludeUnknownNames=True, output_file_path=output_file_path)
        from make_quotes_dataset import split_csv_by_pauses
        split_csv_by_pauses(quotes_output_filePath, quotes_output_filePath)
        columns_to_clean = ['Context', 'Text']  # Columns to clean
        #clean_csv_special_characters(non_quotes_output_filePath, non_quotes_output_filePath, columns_to_clean)
        clean_csv_special_characters(quotes_output_filePath, quotes_output_filePath, columns_to_clean)




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
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

# Define the compute_metrics function for evaluating the model
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Training arguments to customize the training process
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=1000,
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

























