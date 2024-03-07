import pandas as pd
import numpy as np
import torch
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('output.csv')

label_encoder = LabelEncoder()
df['entity_who_said_quote_encoded'] = label_encoder.fit_transform(df['entity_who_said_quote'])

train_df, val_df = train_test_split(df, test_size=0.1)

tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

def tokenize_function(examples):
    text_list = examples['chunk_of_text'].tolist()
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=128)

train_encodings = tokenize_function(train_df)
val_encodings = tokenize_function(val_df)

class QuotationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # Keep tensors on CPU
        item['labels'] = torch.tensor(self.labels[idx])  # Keep labels on CPU
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = QuotationDataset(train_encodings, train_df['entity_who_said_quote_encoded'].values)
val_dataset = QuotationDataset(val_encodings, val_df['entity_who_said_quote_encoded'].values)

model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=len(label_encoder.classes_)).to(device) # Move model to the correct device

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    fp16=True,  # Enable mixed precision training
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

test_cases = val_df['chunk_of_text'].tolist()[:3]
test_encodings = tokenize_function(pd.DataFrame({'chunk_of_text': test_cases}))
test_dataset = QuotationDataset(test_encodings, [0] * len(test_cases))

with torch.no_grad():
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

for text, label in zip(test_cases, predicted_labels):
    print(f"Text: {text}")
    print(f"Predicted label: {label_encoder.inverse_transform([label])[0]}")
    print()
