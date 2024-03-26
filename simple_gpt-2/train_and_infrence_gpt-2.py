
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
dataset = dataset.filter(lambda example: example['formatted_input'] is not None and example['formatted_output'] is not None)

# Function to concatenate the input and output strings
def concatenate_examples(examples):
    return {'text': examples['formatted_input'] + " " + examples['formatted_output']}

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
    formatted_input, formatted_output = pd.read_csv("GPT-2_training_df.csv").sample(1)[["formatted_input", "formatted_output"]].values[0] 
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