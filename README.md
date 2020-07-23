# bert-deid

Code to fine-tune BERT on a medical note de-identification task.

## Install

There are three options for installation:

* Create an environment called `deid` **(recommended)**: `conda env create -f environment.yml`
* conda: `conda install bert_deid`
* pip: `pip install bert_deid`

## Usage

```python
from bert_deid.model import Transformer

# load in a trained model
model_type = 'bert'
model_path = '/data/models/bert-i2b2-2014'
deid_model = Transformer(model_type, model_path)

text = 'Dr. Somayah says I have had a pneumothorax since 2019-01-01.'
print(deid_model.apply(text, repl='___'))

# we can also get the original predictions
preds, lengths, offsets = deid_model.predict(text)

# print out the identified entities
for p in range(preds.shape[0]):
    start, stop = offsets[p], offsets[p] + lengths[p]

    # most likely prediction
    idxMax = preds[p].argmax()
    label = deid_model.label_set.id_to_label[idxMax]
    print(f'{text[start:stop]:15s} {label}')
```

## Training and evaluating a transformer model

First, you'll need a suitable dataset. Right now this can be: i2b2_2014, i2b2_2006, PhysioNet, or Dernoncourt-Lee.
A dataset is considered suitable if it is saved in the right format. Dataset formats are as follows:

* a root folder dedicated to the dataset
* train/test subfolders
* each train/test subfolder has ann/txt subfolders
* the txt subfolder has files with the `.txt` extension containing the text to be deidentified
* the ann subfolder has files with the `.gs` extension containing a CSV of gold standard de-id annotations

Here's an example:

```
i2b2_2014
├── train
│   ├── ann
│   │   ├── 100-01.gs
│   │   ├── 100-02.gs
│   │   └── 100-03.gs
│   └── txt
│       ├── 100-01.txt
│       ├── 100-02.txt
│       └── 100-03.txt
└── test
    ├── ann
    │   ├── 110-01.gs
    │   ├── 110-02.gs
    │   └── 110-03.gs
    └── txt
        ├── 110-01.txt
        ├── 110-02.txt
        └── 110-03.txt
```

With the dataset available, create the environment:

`conda create env -f environment.yml`

Activate the environment:

`conda activate deid`

Train a model (e.g. BERT):

```sh
python scripts/train_transformer.py --data_dir /data/deid-gs/i2b2_2014 --data_type i2b2_2014 --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --output_dir /data/models/bert-model-i2b2-2014 --do_train --overwrite_output_dir
```

Note this will only use data from the `train` subfolder of the `--data_dir` arg. Once the model is trained it can be used as above.
