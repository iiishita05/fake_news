from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import torch
from datasets import Dataset

MODEL_NAME = "bert-base-uncased"  # Change to "roberta-base" or "distilbert-base-uncased" as needed

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['text', 'label']]
    return Dataset.from_pandas(df)

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def train_model():
    dataset = load_data("../datasets/processed_data.csv")
    dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="../models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="../logs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # Use the same for simplicity
    )

    trainer.train()
    trainer.save_model(f"../models/{MODEL_NAME}")

if __name__ == "__main__":
    train_model()
