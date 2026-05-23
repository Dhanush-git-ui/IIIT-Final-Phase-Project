import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

# Sample dataset
texts = [
    "I am very happy today",
    "I feel sad",
    "Leave me alone",
    "I am excited",
    "I hate this",
    "This is wonderful",
    "I feel terrible",
    "I am scared",
    "I love this",
    "I am angry"
]

labels = [
    "happy",
    "sad",
    "angry",
    "happy",
    "angry",
    "happy",
    "sad",
    "fear",
    "happy",
    "angry"
]

# Create dataframe
df = pd.DataFrame({
    "text": texts,
    "label": labels
})

# Encode labels
encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["label"])

# Train test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

# Tokenization function
def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True
    )

# Create datasets
train_dataset = Dataset.from_dict({
    "text": train_texts.tolist(),
    "label": train_labels.tolist()
})

test_dataset = Dataset.from_dict({
    "text": test_texts.tolist(),
    "label": test_labels.tolist()
})

# Tokenize
train_dataset = train_dataset.map(tokenize, batched=True)

test_dataset = test_dataset.map(tokenize, batched=True)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(encoder.classes_)
)

import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Extract and save training logs
history = trainer.state.log_history
train_logs = [log for log in history if 'loss' in log]
eval_logs = [log for log in history if 'eval_loss' in log]

train_epochs = [log['epoch'] for log in train_logs]
train_losses = [log['loss'] for log in train_logs]

eval_epochs = [log['epoch'] for log in eval_logs]
eval_losses = [log['eval_loss'] for log in eval_logs]
eval_accs = [log['eval_accuracy'] for log in eval_logs if 'eval_accuracy' in log]

os.makedirs("../../Results/plots", exist_ok=True)

# Save loss plot
plt.figure()
if train_losses:
    plt.plot(train_epochs, train_losses, label='train_loss')
if eval_losses:
    plt.plot(eval_epochs, eval_losses, label='eval_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Text Model Loss')
plt.savefig("../../Results/plots/text_loss.png")
plt.close()

# Save accuracy plot
plt.figure()
if eval_accs:
    plt.plot(eval_epochs, eval_accs, label='eval_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Text Model Accuracy')
plt.savefig("../../Results/plots/text_accuracy.png")
plt.close()

# Save model
model.save_pretrained("bert_emotion_model")

print("BERT Emotion Model Trained Successfully!")