from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

MODEL_NAME = "bert-base-uncased"

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def evaluate_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(f"../models/{MODEL_NAME}")

    df = load_data("../datasets/processed_data.csv")

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    correct = 0
    total = len(texts)

    for text, label in zip(texts, labels):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()
        if predicted_label == label:
            correct += 1

    accuracy = correct / total
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
