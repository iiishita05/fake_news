from transformers import DistilBertForSequenceClassification

def load_distilbert_model(num_labels=2):
    return DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
