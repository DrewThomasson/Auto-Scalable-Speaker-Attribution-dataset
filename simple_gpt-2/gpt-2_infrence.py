
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





