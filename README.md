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
export MODEL_DIR="~/bert_deid_model/"
bert_deid download
```

## Usage (Shell)

From the command line, we can call `bert_deid` to apply it to any given text:

```sh
export MODEL_DIR="~/bert_deid_model/"
bert_deid apply --text "hello dr. somayah"
```

Text can also be piped to `bert_deid`. Alternatively, the `--text_dir` argument allows running the package on all files in a folder:

```sh
mkdir tmp
echo "hello dr. somayah" > tmp/example1.txt
echo "No pneumothorax since 2019-01-01." > tmp/example2.txt
bert_deid apply --text_dir tmp
```

Deidentified files are output with the `.deid` extension, e.g. `tmp/example1.txt` would become `tmp/example1.txt.deid`.

## Usage (Python)

The model can also be imported and used directly within Python.

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

The `binary_evaluation.py` script can be used to assess performance on a test set. First, we'll need to generate the predictions:

```sh
export TEST_SET_PATH='/enc_data/deid-gs/i2b2_2014/test'
export MODEL_PATH='/enc_data/models/bert-i2b2-2014'
export PRED_PATH='out/'

python scripts/output_preds.py --data_dir ${TEST_SET_PATH} --model_dir ${MODEL_PATH} --output_folder ${PRED_PATH}
```

This outputs the predictions to the `out` folder. If we look at one of the files, we can see each prediction is a CSV of stand-off annotations. Here are the top few lines from the `110-01.pred` file:

```
document_id,annotation_id,start,stop,entity,entity_type,comment
110-01,4,16,20,2069,DATE,
110-01,5,20,21,-,DATE,
110-01,6,21,23,04,DATE,
110-01,7,23,24,-,DATE,
110-01,8,24,26,07,DATE,
```

We can now evaluate the predictions using the ground truth:

```sh
python scripts/binary_evaluation.py --pred_path ${PRED_PATH} --text_path ${TEST_SET_PATH}/txt --ref_path ${TEST_SET_PATH}/ann
```

For our trained model, this returned:

* Macro Se: 0.9818
* Macro P+: 0.9885
* Macro F1: 0.9840
* Micro Se: 0.9816
* Micro P+: 0.9892
* Micro F1: 0.9854

We can also look at individual predictions for a given file:

```sh
export FN=110-02
python scripts/print_annotation.py -p ${PRED_PATH}/${FN}.pred -t ${TEST_SET_PATH}/txt/${FN}.txt -r ${TEST_SET_PATH}/ann/${FN}.gs
```

If we would like a multi-class evaluation, we need to know about any label transformations done by the model, so we call a different script:

```sh
python scripts/eval.py --model_dir ${MODEL_PATH} --pred_path ${PRED_PATH} --text_path ${TEST_SET_PATH}/txt --ref_path ${TEST_SET_PATH}/ann
```
