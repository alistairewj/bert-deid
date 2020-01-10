import os
import pickle
from pathlib import Path
from bisect import bisect_left, bisect_right
import logging

import numpy as np
from transformers import BertTokenizer
from bert_deid.model import Transformer
from bert_deid.processors import load_labels
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# load in a trained model
model_type = 'bert'  # -base-uncased'
model_path = '/enc_data/models/bert-i2b2-2014'
transformer = Transformer(
    model_type,
    model_path,
    max_seq_length=128,
    task_name='i2b2_2014',
    cache_dir=None,
    device='cpu'
)
label_to_id = {v: k for k, v in transformer.label_id_map.items()}

data_path = Path('/enc_data/deid-gs/i2b2_2014/test')

files = os.listdir(data_path / 'txt')
files = [f for f in files if f.endswith('.txt')]
data = []

preds = None
lengths = []
offsets = []
labels = []

for f in tqdm(files, total=len(files), desc='Files'):
    with open(data_path / 'txt' / f, 'r') as fp:
        text = ''.join(fp.readlines())

    ex_preds, ex_lengths, ex_offsets = transformer.predict(text)

    if preds is None:
        preds = ex_preds
    else:
        preds = np.append(preds, ex_preds, axis=0)

    lengths.append(ex_lengths)
    offsets.append(ex_offsets)

    # load in gold standard
    gs_fn = data_path / 'ann' / f'{f[:-4]}.gs'
    gs = load_labels(gs_fn)

    # convert labels to align with preds
    gs = sorted(gs, key=lambda x: x[1])
    label_tokens = ['O'] * len(ex_offsets)

    for i, g in enumerate(gs):
        idxStart = bisect_left(ex_offsets, g[1])
        idxStop = bisect_right(ex_offsets, g[1] + g[2])
        label_tokens[idxStart:idxStop] = [g[0]] * (idxStop - idxStart)

    # convert label tokens to label_ids
    label_tokens = [label_to_id[l] for l in label_tokens]
    labels.extend(label_tokens)

with open('preds.pkl', 'wb') as fp:
    pickle.dump([files, preds, labels, lengths, offsets], fp)