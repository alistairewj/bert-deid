# bert-deid

Code to fine-tune BERT on a medical note de-identification task.

Install:

```
pip install -e .
```

This is an "editable" or developer install; it symlinks the directory so changes to the module are reflected instantly in the environment.

## Training and evaluating a transformer model

First, you'll need a suitable dataset. Right now this can be: i2b2_2014, i2b2_2006, PhysioNet, or Dernoncourt-Lee.
A dataset is considered suitable if it is saved in the right format. Dataset formats are as follows:

* a folder dedicated to the dataset
* train/test subfolders
* each train/test subfolder has ann/txt subfolders
* the txt subfolder has files with the `.txt` extension containing the text to be deidentified
* the ann subfolder has files with the `.gs` extension containing a CSV of gold standard de-id annotations

Look at the `deid-gs` repository for examples of datasets in this format.

With the dataset available, create the environment:

`conda create env -f environment.yml`

Activate the environment:

`conda activate deid`

Train a model (e.g. BERT):

```sh
python scripts/train_transformer.py --data_dir /data/deid-gs/i2b2_2014 --data_type i2b2_2014 --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --output_dir /data/models/bert-model-i2b2-2014 --do_train --overwrite_output_dir
```

Note this will only use data from the `train` subfolder of the `--data_dir` arg.
Now we can evaluate the model:

```python
from transformers import BertTokenizer
from bert_deid.model import Transformer

# load in a trained model
model_type = 'bert'
model_path = '/data/models/bert-i2b2-2014'
transformer = Transformer(
    model_type,
    model_path,
    max_seq_length=128,
    task_name='i2b2_2014',
    cache_dir=None,
    device='cpu'
)


text = 'Dr. Michael Bolton says I have had a pneumothorax since 2019-01-01.'

preds, lengths, offsets = transformer.predict(text)

# print out the predictions
for p in range(preds.shape[0]):
    start, stop = offsets[p], offsets[p] + lengths[p]

    # most likely prediction
    idxMax = preds[p].argmax()
    label = transformer.label_id_map[idxMax]
    print(f'{text[start:stop]:15s} {label}')
```