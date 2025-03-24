import matplotlib.pyplot as plt
from train import train_model
from data_processing import load_dataset
from evaluate import evaluate_model

DATASETS = {
    "LIAR": "datasets/liar.csv",
    "FakeNewsNet": "datasets/fakenewsnet.csv",
    "COVID-19": "datasets/covid19_fake.csv",
    "ISOT": "datasets/isot.csv",
    "FEVER": "datasets/fever.csv"
}

MODELS = ["bert-base-uncased", "roberta-base", "distilbert-base"]

results = {}

for model_name in MODELS:
    for dataset_name, path in DATASETS.items():
        print(f"Training {model_name} on {dataset_name}...")
        df = load_dataset(path)
        model = train_model(dataset_name, df)
        trainer = model["trainer"]
        test_dataset = model["test_dataset"]
        test_labels = model["test_labels"]
        
        metrics = evaluate_model(trainer, test_dataset, test_labels)
        results[(model_name, dataset_name)] = metrics

plt.figure(figsize=(10, 5))
for model_name in MODELS:
    accuracy_scores = [results[(model_name, d)]["Accuracy"] for d in DATASETS.keys()]
    plt.plot(DATASETS.keys(), accuracy_scores, marker='o', label=model_name)

plt.xlabel("Datasets")
plt.ylabel("Accuracy")
plt.title("Fake News Detection Model Benchmarking")
plt.legend()
plt.xticks(rotation=15)
plt.show()