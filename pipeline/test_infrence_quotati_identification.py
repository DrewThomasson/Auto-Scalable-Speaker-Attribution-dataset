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
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizerFast
from tqdm import tqdm
import tkinter as tk
from tkinter import scrolledtext

# Define the replacement functions
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
    # Initialize the BERT tokenizer
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
    return label_encoder[predicted_label] == "Quote"

def fill_is_quote_column(df, model_checkpoint_path="./quotation_identifer_model/checkpoint-1000"):
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

def transfer_quotes(complete_df, incomplete_df):
    # Transfers quotes from the complete DataFrame to the incomplete DataFrame.
    for index, row in complete_df.iterrows():
        is_quote = row['Is Quote']
        if pd.notna(is_quote):
            incomplete_df.at[index, 'Is Quote'] = is_quote
    return incomplete_df

def visualize_quotes(df):
    root = tk.Tk()
    root.title("Text Visualization")

    text_box = scrolledtext.ScrolledText(root, width=40, height=10, wrap=tk.WORD)
    text_box.grid(row=0, column=0, padx=10, pady=10)

    def is_actual_quote(start_idx, end_idx, df):
        quote_delimiters = get_quote_delimiters(df)
        for left_quote_start, left_quote_end, right_quote_start, right_quote_end in quote_delimiters:
            if start_idx >= left_quote_start and end_idx <= right_quote_end:
                return True
        return False

    def get_quote_delimiters(df):
        quote_delimiters = []
        quote_delimiter_pattern = re.compile(r'[“”"‘’\'\']')
        for _, row in df.iterrows():
            context = row['Context']
            matches = quote_delimiter_pattern.finditer(context)
            quotes = [match.span() for match in matches]
            for i in range(0, len(quotes), 2):
                if i + 1 < len(quotes):
                    quote_delimiters.append((quotes[i][0], quotes[i][1], quotes[i+1][0], quotes[i+1][1]))
        return quote_delimiters

    # Calculate the number of correct and incorrect predictions
    total_segments = len(df)
    correct_predictions = 0
    incorrect_predictions = 0

    for _, row in df.iterrows():
        is_quote_prediction = row['Is Quote']
        actual_quote = is_actual_quote(row['Text start char'], row['Text end char'], df)
        if is_quote_prediction == actual_quote:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    #accuracy = (correct_predictions / total_segments) * 100

    # Display the accuracy
    #accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_segments})")
    #accuracy_label.grid(row=1, column=0, padx=10, pady=10)

    # Function to highlight text based on 'Is Quote' column
    def highlight_text():
        text_box.delete('1.0', tk.END)
        red_highlight_count = 0
        yellow_highlight_count = 0
        no_highlight_count = 0
        for _, row in df.iterrows():
            text = row['Text']
            is_quote = row['Is Quote']
            start_idx = row['Text start char']
            end_idx = row['Text end char']
            if is_quote and not is_actual_quote(start_idx, end_idx, df):
                text_box.insert(tk.END, text + "\n", 'wrong')
                red_highlight_count += 1
            elif is_quote:
                text_box.insert(tk.END, text + "\n", 'quote')
                yellow_highlight_count += 1
            else:
                text_box.insert(tk.END, text + "\n")
                no_highlight_count += 1
        # Print the counts of each highlight category
        print(f"Number of incorrect highlights (red): {red_highlight_count}")
        print(f"Number of correct highlights (yellow): {yellow_highlight_count}")
        print(f"Number of non-highlighted segments: {no_highlight_count}")
        # Display the accuracy
        total_segments = red_highlight_count+yellow_highlight_count+no_highlight_count
        correct_predictions = yellow_highlight_count+no_highlight_count
        accuracy = (correct_predictions / total_segments) * 100
        accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_segments})")
        accuracy_label.grid(row=1, column=0, padx=10, pady=10)

    text_box.tag_configure('quote', background='yellow')
    text_box.tag_configure('wrong', underline=True)  # Only underline for incorrect predictions

    highlight_text()  # Initial highlight

    root.mainloop()





# Main process
df = Process_txt_into_BERT_quotes_input_dataframe('Book.txt')
df = fill_is_quote_column(df)
original_df = df.copy()  # Copy the dataframe to preserve the original data
original_df = transfer_quotes(df, original_df)
visualize_quotes(original_df)
