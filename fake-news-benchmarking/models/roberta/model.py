from transformers import RobertaForSequenceClassification

def load_roberta_model(num_labels=2):
    return RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
