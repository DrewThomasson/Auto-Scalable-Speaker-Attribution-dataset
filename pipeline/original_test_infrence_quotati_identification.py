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



def Process_txt_into_BERT_quotes_input_dataframe(filepath):
    
    # Read the entire text file
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Pre-process the text for sentence splitting
    processed_text = replace_titles_and_abbreviations(text)
    
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
        context_start = max(0, start_idx - 200)  # Assuming 200 characters represent 200 BERT tokens
        context_end = min(len(text), end_idx + 200)
        
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
            context_start_token_idx = max(0, start_token_idx - 200)
            context_end_token_idx = min(len(tokenized_text), end_token_idx + 200)
            
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
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re


def predict_quote(context, text, model_checkpoint_path="./quotation_identifer_model/checkpoint-1000"):

    # Combine context and text to create the formatted input
    formatted_input = f"{context} : Is Sentence Quote : {text}"
    
    # Load the pre-trained model checkpoint
    model = DistilBertForSequenceClassification.from_pretrained(model_checkpoint_path)
    
    # Define the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
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




def fill_is_quote_column(df, model_checkpoint_path="./quotation_identifer_model/checkpoint-1000"):
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

def fill_is_quote_column(df, model_checkpoint_path="./quotation_identifer_model/checkpoint-1000"):
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