#This is the code to train the big bird model on the dataset
import pandas as pd
import numpy as np
import torch
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

# Set device to GPU if available, else CPU
import torch

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

print(f"Using device: {device}")

print(f"Using device: {device}")

df = pd.read_csv('output_dir/all_books_processed_data.csv')

# Format the input as "{chunk_of_text} : quote in text {specific_quote}"
df['formatted_input'] = df['chunk_of_text'] + " : quote in text " + df['specific_quote']

label_encoder = LabelEncoder()
df['entity_who_said_quote_encoded'] = label_encoder.fit_transform(df['entity_who_said_quote'])

train_df, val_df = train_test_split(df, test_size=0.1)

tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

# Adjust tokenize_function to use the new formatted input
def tokenize_function(examples):
    text_list = examples['formatted_input'].tolist()
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=128)

train_encodings = tokenize_function(train_df)
val_encodings = tokenize_function(val_df)

class QuotationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = QuotationDataset(train_encodings, train_df['entity_who_said_quote_encoded'].values)
val_dataset = QuotationDataset(val_encodings, val_df['entity_who_said_quote_encoded'].values)

model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=len(label_encoder.classes_)).to(device)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}
if device.type == "cuda":
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        fp16=True,  # Enable mixed precision only on CUDA devices
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
else:
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

final_metrics = trainer.evaluate()
print(final_metrics)

# Use the formatted input for test prediction examples
test_cases_formatted = val_df['formatted_input'].tolist()[:3]
test_encodings = tokenize_function(pd.DataFrame({'formatted_input': test_cases_formatted}))
test_dataset = QuotationDataset(test_encodings, [0] * len(test_cases_formatted))

with torch.no_grad():
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

for formatted_input, label in zip(test_cases_formatted, predicted_labels):
    print(f"Formatted Input: {formatted_input}")
    print(f"Predicted Entity: {label_encoder.inverse_transform([label])[0]}")
