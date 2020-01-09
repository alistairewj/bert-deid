from transformers import BertTokenizer
from bert_deid.model import Transformer

# load in a trained model
model_type = 'bert'  # -base-uncased'
model_path = '/data/models/bert-i2b2-2014'
transformer = Transformer(
    model_type,
    model_path,
    token_step_size=100,
    sequence_length=100,
    max_seq_length=128,
    task_name='i2b2_2014',
    cache_dir=None,
    device='cpu'
)

text = 'Dr. Michael Bolton says I have had a pneumothorax since 2019-01-01.'

preds, lengths, offsets = transformer.predict(text)

# print out the predictions
# max_len = np.max(lengths)
for p in range(preds.shape[0]):
    start, stop = offsets[p], offsets[p] + lengths[p]

    # most likely prediction
    idxMax = preds[p].argmax()
    label = transformer.label_id_map[idxMax]
    print(f'{text[start:stop]:15s} {label}')
# labels = [('NAME', 4, 14), ('DATE', 56, 10)]
