from transformers import DistilBertTokenizer

def load_distilbert_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
