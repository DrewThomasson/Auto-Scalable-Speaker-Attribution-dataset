import pandas as pd
import numpy as np
import torch
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import spacy

# Load spaCy model for NER
nlp = spacy.load("en_core_web_md")

# Function to anonymize names in a text
def anonymize_names(text):
    doc = nlp(text)
    anonymized_text = text
    name_map = {}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if ent.text not in name_map:
                name_map[ent.text] = f"NAME_{len(name_map) + 1}"
            anonymized_text = anonymized_text.replace(ent.text, name_map[ent.text])
    return anonymized_text

# Load and preprocess dataset
df = pd.read_csv('output.csv')

# Anonymize names in the 'chunk_of_text' column
df['chunk_of_text'] = df['chunk_of_text'].apply(anonymize_names)

# Preliminary preprocessing
label_encoder = LabelEncoder()
df['entity_who_said_quote_encoded'] = label_encoder.fit_transform(df['entity_who_said_quote'])

# Split the dataset
train_df, val_df = train_test_split(df, test_size=0.1)

# Initialize tokenizer
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

# Tokenization function
def tokenize_function(examples):
    text_list = examples['chunk_of_text'].tolist()
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=128)

# Tokenizing the texts
train_encodings = tokenize_function(train_df)
val_encodings = tokenize_function(val_df)

# Dataset class
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

# Creating datasets
train_dataset = QuotationDataset(train_encodings, train_df['entity_who_said_quote_encoded'].values)
val_dataset = QuotationDataset(val_encodings, val_df['entity_who_said_quote_encoded'].values)

# Model initialization
model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=len(label_encoder.classes_))

# Define compute_metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Training arguments
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

# Start training
trainer.train()

# Evaluate
final_metrics = trainer.evaluate()
print(final_metrics)

# Test cases and predictions, using anonymized text
test_cases = val_df['chunk_of_text'].tolist()[:3]
test_encodings = tokenize_function(pd.DataFrame({'chunk_of_text': test_cases}))
test_dataset = QuotationDataset(test_encodings, [0] * len(test_cases))  # Dummy labels

with torch.no_grad():
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

for text, label in zip(test_cases, predicted_labels):
    print(f"Text: {text}")
    print(f"Predicted label: {label_encoder.inverse_transform([label])[0]}")
    print()
