#Function to be given a txt file and create and return a dataframe with the values 'Text','Context','Text start char','Text end char','Context start char','Context end char','Is Quote'

#Text is the sentence 

#Context is the context of the sentence which contains the sentence being the text + 200 BERT tokens in front of the text  and 200 Bert tokens behind it

#'Text start char' is the start location of the text via characters in refrence to the txt file given

#'Text end char' is the start location of the text via characters in refrence to the txt file given

#'Context start char' is the start location of the text via characters in refrence to the txt file given

#'Context end char' is the start location of the text via characters in refrence to the txt file given

#'Is Quote' Column is left blank for now


#The method for sentence splitting is this methof used here



import pandas as pd
import re

def replace_titles_and_abbreviations(text):
    replacements = {
        r"Mr\.": "<MR>", r"Ms\.": "<MS>", r"Mrs\.": "<MRS>", r"Dr\.": "<DR>",
        r"Prof\.": "<PROF>", r"Rev\.": "<REV>", r"Gen\.": "<GEN>", r"Sen\.": "<SEN>",
        r"Rep\.": "<REP>", r"Gov\.": "<GOV>", r"Lt\.": "<LT>", r"Sgt\.": "<SGT>",
        r"Capt\.": "<CAPT>", r"Cmdr\.": "<CMDR>", r"Adm\.": "<ADM>", r"Maj\.": "<MAJ>",
        r"Col\.": "<COL>", r"St\.": "<ST>", r"Co\.": "<CO>", r"Inc\.": "<INC>",
        r"Corp\.": "<CORP>", r"Ltd\.": "<LTD>", r"Jr\.": "<JR>", r"Sr\.": "<SR>",
        r"Ph\.D\.": "<PHD>", r"M\.D\.": "<MD>", r"B\.A\.": "<BA>", r"B\.S\.": "<BS>",
        r"M\.A\.": "<MA>", r"M\.S\.": "<MS>", r"LL\.B\.": "<LLB>", r"LL\.M\.": "<LLM>",
        r"J\.D\.": "<JD>", r"Esq\.": "<ESQ>",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def revert_titles_and_abbreviations(text):
    replacements = {
        "<MR>": "Mr.", "<MS>": "Ms.", "<MRS>": "Mrs.", "<DR>": "Dr.",
        "<PROF>": "Prof.", "<REV>": "Rev.", "<GEN>": "Gen.", "<SEN>": "Sen.",
        "<REP>": "Rep.", "<GOV>": "Gov.", "<LT>": "Lt.", "<SGT>": "Sgt.",
        "<CAPT>": "Capt.", "<CMDR>": "Cmdr.", "<ADM>": "Adm.", "<MAJ>": "Maj.",
        "<COL>": "Col.", "<ST>": "St.", "<CO>": "Co.", "<INC>": "Inc.",
        "<CORP>": "Corp.", "<LTD>": "Ltd.", "<JR>": "Jr.", "<SR>": "Sr.",
        "<PHD>": "Ph.D.", "<MD>": "M.D.", "<BA>": "B.A.", "<BS>": "B.S.",
        "<MA>": "M.A.", "<MS>": "M.S.", "<LLB>": "LL.B.", "<LLM>": "LL.M.",
        "<JD>": "J.D.", "<ESQ>": "Esq.",
    }
    for placeholder, original in replacements.items():
        text = re.sub(placeholder, original, text)
    return text

def split_text_by_pauses(text):
    text = replace_titles_and_abbreviations(text)
    pattern = r'[.!,;?:]'
    parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
    parts_with_punctuation = [
        part + text[text.find(part) + len(part)]
        if text.find(part) + len(part) < len(text) and text[text.find(part) + len(part)] in '.!,;?'
        else part for part in parts
    ]
    parts_with_punctuation = [revert_titles_and_abbreviations(part) for part in parts_with_punctuation]
    return parts_with_punctuation






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





def split_text_by_pauses(text):
    #replace titles and abbriviations so they dont split first
    replace_titles_and_abbreviations(text)
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

    # Revert titles and abbreviations back to their original forms for each part
    parts_with_punctuation = [revert_titles_and_abbreviations(part) for part in parts_with_punctuation]
    
    
    return parts_with_punctuation

def Process_txt_into_BERT_quotes_input_dataframe(filepath):
    
    # Read the entire text file
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Pre-process the text for sentence splitting
    #should already be done in the split_text_by_pauses function so ignor this i guess
    #processed_text = replace_titles_and_abbreviations(text)
    
    # Split text into sentences
    sentences = split_text_by_pauses(processed_text)
    
    # Initialize DataFrame columns
    data = {
        'Text': [],
        'Context': [],
        'Text start char': [],
        'Text end char': [],
        'Context start char': [],
        'Context end char': [],
        'Is Quote': []  # Leaving blank as instructed
    }
    
    # Iterate over sentences to fill the DataFrame
    for sentence in sentences:
        # Find the original sentence in the text
        start_idx = text.find(sentence)
        end_idx = start_idx + len(sentence)
        
        # Approximate context range - adjust the numbers to match BERT tokens' equivalent
        context_start = max(0, start_idx - 150)  # Assuming 200 characters represent 200 BERT tokens
        context_end = min(len(text), end_idx + 150)
        
        context = text[context_start:context_end]
        
        # Populate data dictionary
        data['Text'].append(sentence)
        data['Context'].append(context)
        data['Text start char'].append(start_idx)
        data['Text end char'].append(end_idx)
        data['Context start char'].append(context_start)
        data['Context end char'].append(context_end)
        data['Is Quote'].append('')  # Blank as per instructions
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df




import pandas as pd
#from transformers import BertTokenizer
from transformers import BertTokenizerFast

def Process_txt_into_BERT_quotes_input_dataframe(filepath):
    # Initialize the BERT tokenizer
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Read the entire text file
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Pre-process the text for sentence splitting
    sentences = split_text_by_pauses(text)
    
    # Initialize DataFrame columns
    data = {
        'Text': [],
        'Context': [],
        'Text start char': [],
        'Text end char': [],
        'Context start char': [],
        'Context end char': [],
        'Is Quote': [],  # Leaving blank as instructed
        'Speaker': []  # Add this line to initialize 'Speaker'
    }
    
    # Tokenize the text to work with tokens directly
    tokenized_text = tokenizer.tokenize(text)
    encoded_text = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded_text['offset_mapping']
    
    # Iterate over sentences to fill the DataFrame
    for sentence in sentences:
        # Find the token index range for the sentence
        start_idx, end_idx = text.find(sentence), text.find(sentence) + len(sentence)
        start_token_idx = next((i for i, offset in enumerate(offsets) if offset[0] == start_idx), None)
        end_token_idx = next((i for i, offset in enumerate(offsets) if offset[1] == end_idx), None)
        
        if start_token_idx is not None and end_token_idx is not None:
            # Define context range in tokens
            context_start_token_idx = max(0, start_token_idx - 150)
            context_end_token_idx = min(len(tokenized_text), end_token_idx + 150)
            
            # Convert token indices back to character indices for context
            context_start_char = offsets[context_start_token_idx][0]
            context_end_char = offsets[min(context_end_token_idx, len(offsets) - 1)][1]
            
            context = text[context_start_char:context_end_char]
            
            # Populate data dictionary
            data['Text'].append(sentence)
            data['Context'].append(context)
            data['Text start char'].append(start_idx)
            data['Text end char'].append(end_idx)
            data['Context start char'].append(context_start_char)
            data['Context end char'].append(context_end_char)
            data['Is Quote'].append('')  # Blank as per instructions
            data['Speaker'].append('')  # Blank as per instructions
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df





#FUCKKKKKK LOOK AT ME DREW POOP
#This will remove any commas from the txt files when it comes to commas in large didgets
#from make_quotes_dataset import process_large_numbers_in_directory
#process_large_numbers_in_directory("books")
from make_quotes_dataset import process_large_numbers_in_txt
process_large_numbers_in_txt('Book.txt')

# You'll need to replace 'your_text_file.txt' with the actual file path
df = Process_txt_into_BERT_quotes_input_dataframe('Book.txt')
print(df)
df.to_csv('BERT_infrence_quote_input.csv', index=False)
#THIS will put all of the values in df into original_df
original_df = df.copy()


#next I want to clean up the two columns "Text" and "Context"


from make_non_quotes_dataset import clean_csv_special_characters
columns_to_clean = ['Context', 'Text']  # Columns to clean
clean_csv_special_characters('BERT_infrence_quote_input.csv', 'BERT_infrence_quote_input.csv', columns_to_clean)


#next I want to make function that will use the pretrained BERT to take in the current dataframe and output with the 'Is Quote' columns rows labled 



import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re


def predict_quote(context, text, model_checkpoint_path="./quotation_identifer_model/checkpoint-3000"):

    # Combine context and text to create the formatted input
    formatted_input = f"{context} : Is Sentence Quote : {text}"
    
    # Load the pre-trained model checkpoint
    model = BertForSequenceClassification.from_pretrained(model_checkpoint_path)
    
    # Define the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the input text
    tokenized_input = tokenizer(formatted_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    # Perform prediction on the input text
    with torch.no_grad():
        outputs = model(**tokenized_input)
    
    # Extract predicted label
    predicted_label = torch.argmax(outputs.logits).item()
    
    # Define label encoder (assuming binary classification)
    label_encoder = {0: "Not a Quote", 1: "Quote"}
    
    # Return True if predicted label is "Quote", False otherwise
    print(label_encoder[predicted_label] == "Quote")
    return label_encoder[predicted_label] == "Quote"




import pandas as pd
from tqdm import tqdm


# Read the CSV file into a DataFrame
csv_df = pd.read_csv('BERT_infrence_quote_input.csv')




def fill_is_quote_column(df, model_checkpoint_path="./quotation_identifer_model/checkpoint-3000"):
    """
    Updates the 'Is Quote' column in the DataFrame based on predictions from the predict_quote function.

    Parameters:
    - df: pandas.DataFrame with at least 'Text' and 'Context' columns.
    - model_checkpoint_path: str, path to the model checkpoint used for prediction.

    Returns:
    - pandas.DataFrame with the 'Is Quote' column updated based on predictions.
    """
    # Ensure 'Is Quote' column exists
    if 'Is Quote' not in df.columns:
        df['Is Quote'] = None
    
    # Iterate over DataFrame rows
    for index, row in df.iterrows():
        context = row['Context']
        text = row['Text']
        # Call the predict_quote function for each row and update the 'Is Quote' column
        df.at[index, 'Is Quote'] = predict_quote(context, text, model_checkpoint_path)
    
    return df




import pandas as pd
from tqdm import tqdm

def fill_is_quote_column(df, model_checkpoint_path="./quotation_identifer_model/checkpoint-3000"):
    """
    Updates the 'Is Quote' column in the DataFrame based on predictions from the predict_quote function.

    Parameters:
    - df: pandas.DataFrame with at least 'Text' and 'Context' columns.
    - model_checkpoint_path: str, path to the model checkpoint used for prediction.

    Returns:
    - pandas.DataFrame with the 'Is Quote' column updated based on predictions.
    """
    # Ensure 'Is Quote' column exists
    if 'Is Quote' not in df.columns:
        df['Is Quote'] = None

    # Initialize tqdm to track progress
    tqdm.pandas(desc="Processing rows", unit="row")

    # Iterate over DataFrame rows with tqdm
    for index, row in tqdm(df.iterrows(), total=len(df)):
        context = row['Context']
        text = row['Text']
        # Call the predict_quote function for each row and update the 'Is Quote' column
        df.at[index, 'Is Quote'] = predict_quote(context, text, model_checkpoint_path)

    return df










csv_df = fill_is_quote_column(csv_df)

csv_df.to_csv('BERT_infrence_quote_input.csv', index=False)





import pandas as pd

def transfer_quotes(complete_df, incomplete_df):
    """
    Transfers quotes from the complete DataFrame to the incomplete DataFrame.

    Parameters:
    - complete_df: pandas.DataFrame with the "Is Quote" column filled.
    - incomplete_df: pandas.DataFrame with the "Is Quote" column not filled.

    Returns:
    - pandas.DataFrame with the "Is Quote" column filled.
    """
    for index, row in complete_df.iterrows():
        is_quote = row['Is Quote']
        if pd.notna(is_quote):
            incomplete_df.at[index, 'Is Quote'] = is_quote

    return incomplete_df

# Example usage
#complete_df = pd.read_csv('complete_dataframe.csv')  # Load your complete DataFrame
#incomplete_df = pd.read_csv('incomplete_dataframe.csv')  # Load your incomplete DataFrame

# Transfer quotes
#incomplete_df_with_quotes = transfer_quotes(complete_df, incomplete_df)

# Save or use the updated incomplete DataFrame
#incomplete_df_with_quotes.to_csv('incomplete_dataframe_with_quotes.csv', index=False)


original_df = transfer_quotes(csv_df, original_df)















import tkinter as tk
from tkinter import scrolledtext
import pandas as pd

def visualize_quotes(df):
    root = tk.Tk()
    root.title("Text Visualization")

    text_box = scrolledtext.ScrolledText(root, width=40, height=10, wrap=tk.WORD)
    text_box.grid(row=0, column=0, padx=10, pady=10)

    # Function to highlight text based on 'Is Quote' column
    def highlight_text():
        text_box.delete('1.0', tk.END)
        for _, row in df.iterrows():
            text = row['Text']
            is_quote = row['Is Quote']
            if is_quote:
                text_box.insert(tk.END, text + "\n", 'quote')
            else:
                text_box.insert(tk.END, text + "\n")

    text_box.tag_configure('quote', background='yellow')

    highlight_text()  # Initial highlight

    root.mainloop()

# Example usage
#csv_df = pd.read_csv('BERT_infrence_quote_input.csv')  # Load your DataFrame here
#visualize_quotes(csv_df)
visualize_quotes(original_df)




"""

#This part will infrence the quoation attribution BERT model
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Assuming csv_df is the DataFrame you've been working with and is still in memory

# Function to load the model from a checkpoint
def load_model_from_checkpoint(checkpoint_path):
    model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    return model

# Dataset class to hold the tokenized inputs
class InferenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Function to perform inference
def predict_speaker(checkpoint_folder, dataframe):
    # Set device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained BERT model
    model = load_model_from_checkpoint(checkpoint_folder).to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare the DataFrame
    dataframe['Speaker'] = "Narrator"  # Default value
    quotes_df = dataframe[dataframe['Is Quote']]  # Filter rows where 'Is Quote' is True
    
    # Tokenizing the data
    formatted_input = quotes_df['Context'] + " : quote in text " + quotes_df['Text']
    encodings = tokenizer(formatted_input.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    dataset = InferenceDataset(encodings)
    
    # Perform prediction
    with torch.no_grad():
        for idx, batch in enumerate(dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=1)
            label = predictions.cpu().numpy()[0]  # Assuming label encoder was used as in the training phase
            # Update the 'Speaker' column based on the prediction
            actual_idx = quotes_df.iloc[idx].name  # Get the actual index from the original DataFrame
            dataframe.at[actual_idx, 'Speaker'] = label  # Here, you'd convert `label` to the actual entity name using the label encoder

    return dataframe




# Example usage
checkpoint_folder = './quotation_attribution_model/checkpoint-200'
# csv_df is your pre-existing DataFrame in memory
updated_csv_df = predict_speaker(checkpoint_folder, original_df)
print(updated_csv_df)






#This will make a visualiation of the ouputs of the BERT quotation attribution model

"""



import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Assuming the checkpoint directory is stored in a variable
checkpoint_dir = './quotation_attribution_model/checkpoint-1000'

# Load the tokenizer and model from the checkpoint
tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
model = BertForSequenceClassification.from_pretrained(checkpoint_dir)

# Assuming the LabelEncoder is saved and needs to be loaded
import joblib
label_encoder = joblib.load('path_to_your_label_encoder.pkl')

class InferenceDataset(Dataset):
    """Dataset class for inference."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def predict_speaker(dataframe, model, tokenizer, label_encoder):
    """Predict speakers and update the DataFrame."""
    dataframe['formatted_input'] = dataframe['Context'] + " : quote in text " + dataframe['Text']
    formatted_inputs = dataframe['formatted_input'].tolist()
    
    encodings = tokenizer(formatted_inputs, padding="max_length", truncation=True, max_length=512)
    dataset = InferenceDataset(encodings)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = []
    with torch.no_grad():
        for item in dataset:
            inputs = {k: v.to(device).unsqueeze(0) for k, v in item.items()}  # Add batch dimension
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(logits.cpu().numpy())
    
    # Convert predictions to labels
    predicted_labels = np.argmax(np.concatenate(predictions), axis=1)
    predicted_speakers = label_encoder.inverse_transform(predicted_labels)
    
    dataframe['Speaker'] = predicted_speakers

# Example usage
df_to_predict = pd.read_csv('path_to_your_dataframe.csv')
predict_speaker(df_to_predict, model, tokenizer, label_encoder)
print(df_to_predict[['Text', 'Context', 'Speaker']])
