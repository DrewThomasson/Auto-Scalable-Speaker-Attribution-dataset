
'''
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
trained_model = GPT2LMHeadModel.from_pretrained("./trained_model")
trained_tokenizer = GPT2Tokenizer.from_pretrained("./trained_model")


def generate_response(input_text):
    input_ids = trained_tokenizer.encode(input_text, return_tensors="pt")
    output = trained_model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = trained_tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split(" ", 1)[1]  # Remove the input text from the response

# Example usage
input_text = "What's your favorite color?"
response = generate_response(input_text)
print(response)




'''






'''

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model

model_path = "./trained_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Function to generate the model's output given an input string
def generate_output(input_string):
    # Tokenize the input string
    input_ids = tokenizer.encode(input_string, return_tensors="pt")

    # Generate the output
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated output
    generated_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_output

# Get the input string from the user
input_string = input("Enter an input string: ")

# Generate the model's output
output_string = generate_output(input_string)

# Print the generated output
print("Generated Output:")
print(output_string)

'''

'''

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model

model_path = "./trained_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Function to generate the model's output given an input string
def generate_output(input_string):
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True, max_length=20)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output = model.generate(
        input_ids,
        max_length=20,  # Further reduce max length if expecting shorter responses
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



# Get the input string from the user
input_string = input("Enter an input string: ")

# Generate the model's output
output_string = generate_output(input_string)

# Print the generated output
print("Generated Output:")
print(output_string)
'''














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



for i in range(5):
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
    print(extract_first_value(output_string))


    #print the correct output
    print("Correct output:")
    formatted_output = formatted_output.replace('<end>', '')
    print(formatted_output)

'''
# Get the input string from the user
input_string = input("Enter an input string: ")

# Generate the model's output
output_string = generate_output(input_string)

# Print the generated output
print("Generated Output:")
print(output_string)

'''
