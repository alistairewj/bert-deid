from transformers import BertTokenizer
from bert_deid.processors import tokenize_with_labels, InputExample

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True
)

text = 'I have a pneumothorax, but the doctors say they are working on it. Luckily it is not a pleural effusion! Eh?'
labels = [('d', 9, 12), ('d', 87, 103)]

example = InputExample('1', text, labels=labels)

tokens, offsets, labels = tokenize_with_labels(
    tokenizer, example, pad_token_label_id=-100
)

assert len(offsets) == len(tokens)
assert [l == 'O' for l in labels[0:3]]
assert [l == 'd' for l in labels[3:9]]