from transformers import RobertaTokenizer

def load_roberta_tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-base")
