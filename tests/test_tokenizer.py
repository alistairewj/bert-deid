from transformers import BertTokenizer, AlbertTokenizer
from bert_deid.tokenization import tokenize_with_labels
from bert_deid.processors import InputExample


def test_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True
    )

    text = 'I have a pneumothorax, but the doctors say they are working on it. Luckily it is not a pleural effusion! Eh?'
    labels = [('d', 9, 12), ('d', 87, 103)]

    example = InputExample('1', text, labels=labels)

    tokens, token_labels, token_sw, offsets, lengths = tokenize_with_labels(
        tokenizer, example, pad_token_label_id=-100
    )

    assert len(offsets) == len(tokens)
    assert all([l == 'O' for l in token_labels[0:3]])
    assert all([l == 'O' for l in token_labels[9:25]])
    # ensure we have mapped 'D' labels correctly
    assert token_labels[3] == 'D'
    assert all([l == -100 for l in token_labels[4:9]])
    assert token_labels[25] == 'D'
    assert token_labels[28] == 'D'
    assert all([l == -100 for l in token_labels[26:28]])
    assert all([l == -100 for l in token_labels[29:31]])


def test_albert_tokenizer():
    do_lower_case = True
    tokenizer = AlbertTokenizer.from_pretrained(
        'albert-base-v1', do_lower_case=do_lower_case
    )

    text = 'I have a pneumothorax, but the doctors say they are working on it. Luckily it is not a pleural effusion! Eh?'
    labels = [('D', 9, 12), ('D', 87, 103)]

    example = InputExample('1', text, labels=labels)

    tokens, token_labels, token_sw, offsets, lengths = tokenize_with_labels(
        tokenizer, example, pad_token_label_id=-100
    )

    assert len(offsets) == len(tokens)

    # check that the offsets correctly capture the text
    for i, token in enumerate(tokens):
        if offsets[i] < 0:
            continue

        token = tokens[i]
        if not token_sw[i]:
            token = token[1:]

        if do_lower_case:
            assert text[offsets[i]:offsets[i] + lengths[i]].lower() == token
        else:
            assert text[offsets[i]:offsets[i] + lengths[i]] == token

    assert all([l == 'O' for l in token_labels[0:3]])
    assert all([l == 'O' for l in token_labels[8:17]])
    assert all([l == 'O' for l in token_labels[18:23]])
    # albert assigns punctuation as a subword
    assert tokens[17] == '.'
    assert token_labels[17] == -100

    # ensure we have mapped 'd' labels correctly
    assert token_labels[3] == 'D'
    assert all([l == -100 for l in token_labels[4:8]])
    assert token_labels[23] == 'D'
    assert token_labels[24] == -100
    assert token_labels[25] == 'D'
    assert token_labels[26] == -100