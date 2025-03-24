import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAMES = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased"
]

def benchmark_model(model_name, text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()
    
    return end_time - start_time

def run_benchmark():
    text = "The government has announced new policies to fight climate change."

    results = {}
    for model_name in MODEL_NAMES:
        print(f"Benchmarking {model_name}...")
        exec_time = benchmark_model(model_name, text)
        results[model_name] = exec_time

    print("\nBenchmark Results (Inference Time in Seconds):")
    for model, time_taken in results.items():
        print(f"{model}: {time_taken:.4f}s")

if __name__ == "__main__":
    run_benchmark()
