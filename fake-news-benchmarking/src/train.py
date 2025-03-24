import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_dataset(df):
    return tokenizer(list(df['clean_text']), padding=True, truncation=True, max_length=512, return_tensors="pt")

def train_model(dataset_name, df):
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['clean_text'], df['label'], test_size=0.2)

    train_encodings = tokenize_dataset(pd.DataFrame({'clean_text': train_texts}))
    test_encodings = tokenize_dataset(pd.DataFrame({'clean_text': test_texts}))

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": list(train_labels)
    })

    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": list(test_labels)
    })

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./models/{dataset_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir=f"./logs/{dataset_name}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    return model