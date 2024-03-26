
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