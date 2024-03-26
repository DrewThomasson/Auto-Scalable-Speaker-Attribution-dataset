
'''
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the CSV file into a DataFrame
df = pd.read_csv("test.csv")

# Combine input and output strings
train_data = df["Input_string"] + " " + df["Output_string"]

# Save the combined data to a text file
train_file = "train_data.txt"
train_data.to_csv(train_file, index=False, header=False)

# Load the pre-trained DistilGPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Prepare the dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
    overwrite_cache=True,
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=3,
    save_total_limit=1,
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./trained_model")





'''









'''

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load the CSV file into a DataFrame
df = pd.read_csv("test.csv")

# Combine input and output strings
train_data = df["Input_string"] + " " + df["Output_string"]

# Save the combined data to a text file
train_file = "train_data.txt"
train_data.to_csv(train_file, index=False, header=False)

# Load the pre-trained DistilGPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device) 
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Prepare the dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
    overwrite_cache=True,
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=1,
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer to the "trained_model" folder
model_save_path = "./trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

'''























import pandas as pd
from datasets import Dataset
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the CSV file into a DataFrame
df = pd.read_csv("test.csv")

# Create a Hugging Face Dataset from the DataFrame
dataset = Dataset.from_pandas(df)

# Filter out rows where either 'Input_string' or 'Output_string' is missing
dataset = dataset.filter(lambda example: example['Input_string'] is not None and example['Output_string'] is not None)

# Function to concatenate the input and output strings
def concatenate_examples(examples):
    return {'text': examples['Input_string'] + " " + examples['Output_string']}

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
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set the format to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1999,
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
