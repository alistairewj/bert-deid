# bert-deid

Code to fine-tune BERT on a medical note de-identification task.

## Install

There are three options for installation:

* Create an environment called `deid` **(recommended)**: `conda env create -f environment.yml`
* conda: `conda install bert_deid`
* pip: `pip install bert_deid`

## Usage

```python
from bert_deid import model as bert_deid_model

# Load a trained model
bert_model = bert_deid_model.BertForDEID(
    model_dir="models/i2b2_2014"
)

# Apply that model on some text
text = 'Discussed clinical course with Dr. Somayah yesterday.'
text_deid = bert_model.deid(text)
print(text_deid)

# Show the annotations
ann = bert_model.annotate(text)
print(ann)
```

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


python scripts/calculate_performance.py --pred_path /data/models/predictions/simple --text_path /data/deid-gs/i2b2_2014/test/txt --ref_path /data/deid-gs/i2b2_2014/test/ann --stats_path i2b2_2014_stats.csv --task i2b2_2014 --label_transform simple --tokens_path i2b2_2014_comparison.csv --splitter "\\s"

## Evaluation

In order to compare to previous work, it's important to tokenize entities in a similar fashion.
Dernoncourt and Lee et al. use [Stanford's CoreNLP tokenizer](https://github.com/stanfordnlp/python-stanford-corenlp) and [spaCy's tokenizer](https://spacy.io/usage#installation).
Try this to install:

```sh
# install spacy
conda install -c conda-forge spacy
# download the model needed
python -m spacy download en_core_web_sm

# install stanford NLP
pip install stanfordnlp
# download the necessary resources
python -c "import stanfordnlp; stanfordnlp.download('en', force=True)"
```

For further detail on installation and troubleshooting, [head over to their GitHub repository](https://github.com/stanfordnlp/python-stanford-corenlp).



## PhysioNet Google annotations



```python

# on quadro

CUDA_VISIBLE_DEVICES=0 python -m pdb scripts/output_preds.py --data_dir /data/deid-gs/i2b2_2014/test --model_dir /data/models/bert-base-uncased-i2b2-2014 --model_type bert --task i2b2_2014 --output_folder i2b2_2014_test_output



CUDA_VISIBLE_DEVICES=0 python -m pdb scripts/eval_pred.py --pred_path i2b2_2014_test_output --text_path /data/deid-gs/i2b2_2014/test/txt --ref_path /data/deid-gs/i2b2_2014/test/ann

```



```python

# on quadro

CUDA_VISIBLE_DEVICES=0 python -m pdb scripts/output_preds.py --data_dir /data/deid-gs/physionet_google/data --model_dir /data/models/bert-base-uncased-i2b2-2014 --model_type bert --task i2b2_2014 --output_folder physionet_google_output



CUDA_VISIBLE_DEVICES=0 python -m pdb scripts/eval_pred.py --pred_path physionet_google_output --text_path /data/deid-gs/physionet_google/data/txt --ref_path /data/deid-gs/physionet_google/data/ann

```


