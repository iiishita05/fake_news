from transformers import BertForSequenceClassification

def load_bert_model(num_labels=2):
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
