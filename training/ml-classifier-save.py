# %% Training Script (train_model.py)
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import joblib

# Load dataset
df = pd.read_csv('english.csv')

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X = df['text'].tolist()
y = df['label'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

train_encodings = tokenizer(
    X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True,
                           padding=True, max_length=128)

# Dataset preparation


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CustomDataset(train_encodings, y_train)
test_dataset = CustomDataset(test_encodings, y_test)

# Define model and training arguments
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train and evaluate model
trainer.train()
trainer.evaluate()

# Save the trained model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Save datasets using joblib
joblib.dump(train_encodings, './saved_model/train_encodings.joblib')
joblib.dump(test_encodings, './saved_model/test_encodings.joblib')
joblib.dump(y_train, './saved_model/y_train.joblib')
joblib.dump(y_test, './saved_model/y_test.joblib')

print("Model, tokenizer, and encodings saved successfully!")
