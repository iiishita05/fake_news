from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(trainer, test_dataset, test_labels):
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)

    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}