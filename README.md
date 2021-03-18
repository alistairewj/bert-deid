# bert-deid

Code to fine-tune BERT on a medical note de-identification task.

## Install

* **(Recommended)** Create an environment called `deid`
    * `conda env create -f environment.yml`
<!-- * conda: `conda install bert-deid` -->
* pip install locally
    * `pip install bert-deid`

## Download

To download the model, we have provided a helper script in bert-deid:

```sh
# note: MODEL_DIR environment variable used by download
# by default, we download to bert_deid_model in the current directory
export MODEL_DIR="bert_deid_model"
bert_deid download
```

## Usage

The model can be imported and used directly within Python.

```python
from bert_deid.model import Transformer

# load in a trained model
model_path = 'bert_deid_model'
deid_model = Transformer(model_path)

with open('tests/example_note.txt', 'r') as fp:
    text = ''.join(fp.readlines())

print(deid_model.apply(text, repl='___'))

# we can also get the original predictions
preds = deid_model.predict(text)

# print out the identified entities
for p, pred in enumerate(preds):
    prob = pred[0]
    label = pred[1]
    start, stop = pred[2:]

    # print the prediction labels out
    print(f'{text[start:stop]:15s} {label} ({prob:0.3f})')
```