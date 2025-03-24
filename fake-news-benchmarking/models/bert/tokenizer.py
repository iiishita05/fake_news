from transformers import BertTokenizer

def load_bert_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")
