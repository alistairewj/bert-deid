# bert-deid

Code to fine-tune BERT on a medical note de-identification task.


## Quickstart

CoNLL:

`python bert_ner.py --data_dir conll --bert_model bert-base-cased --task_name conll --output_dir conll_model --do_train --do_eval`

i2b2 2014:

`CUDA_VISIBLE_DEVICES=0 python bert_ner.py --data_dir i2b2_2014 --bert_model bert-base-uncased --task_name i2b2 --output_dir i2b2_2014_model --max_seq_length=128 --do_train --train_batch_size 128 --num_train_epochs 5 --warmup_proportion=0.4 --do_eval --output_predictions --eval_batch_size 128`

dernoncourt:

`CUDA_VISIBLE_DEVICES=0 python bert_ner.py --data_dir dernoncourt --bert_model bert-base-cased --task_name i2b2 --output_dir dernoncourt_model_cased --max_seq_length=128 --do_train --train_batch_size 128 --num_train_epochs 5 --warmup_proportion=0.4`

### What we actually run on i2b2_2014

This code should work out of the box if you have `deid-gs` and `bert-deid` in the same folder, as relative symlinks are used from here to the `deid-gs` repo. Prep symlinks:

```
ln -s ../deid-gs/utils
```

Generate training set with overlap -> train.csv, and test set with *no* overlap -> test.csv

```
python create_csv.py -i ../deid-gs/i2b2_2014/train -o data/i2b2_2014/train.csv --bert_model bert-base-cased -g -m overlap --sequence-length 100 --step-size 20
```


train model:

`CUDA_VISIBLE_DEVICES=0 python bert_ner.py --data_dir data/i2b2_2014 --bert_model bert-base-cased --task_name i2b2 --output_dir models/i2b2_2014_overlap_cased --max_seq_length=128 --do_train --train_batch_size 32 --num_train_epochs 3 --warmup_proportion=0.4 --seed 7841`

evaluate model (generating predictions):

`CUDA_VISIBLE_DEVICES=0 python apply_model.py --model_dir models/i2b2_2014_overlap_cased --input ../deid-gs/i2b2_2014/test/txt --labels ../deid-gs/i2b2_2014/test/ann --bert_model bert-base-cased --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --conll`

predictions are in `preds.conll`

`./evaluate/conlleval -r < preds.conll`

### bert-big

```
CUDA_VISIBLE_DEVICES=0 python bert_ner.py --data_dir data/i2b2_2014 --bert_model bert-large-cased --task_name i2b2 --output_dir models/i2b2_2014_overlap_large --max_seq_length=128 --do_train --train_batch_size 32 --num_train_epochs 3 --warmup_proportion=0.4 --seed 7802;
```

uncased:


CUDA_VISIBLE_DEVICES=0 python bert_ner.py --data_dir data/i2b2_2014 --bert_model bert-base-uncased  --task_name i2b2 --output_dir models/i2b2_2014_overlap_uncased --max_seq_length=128 --do_train --do_lower_case--train_batch_size 32 --num_train_epochs 3 --warmup_proportion=0.4 --seed 7801;

## Quick eval

dernoncourt:

`CUDA_VISIBLE_DEVICES=0 python bert_ner.py --data_dir dernoncourt_overlap --bert_model bert-base-cased --task_name i2b2 --output_dir dernoncourt_overlap_model_cased --max_seq_length=128 --do_eval`


## Other commands

### UNCASED
for dataset in i2b2_2006 physionet_goldstandard dernoncourt_goldstandard;
do python apply_model.py --model_dir models/${dataset}_uncased --input ../deid-gs/${dataset}/test/txt --labels ../deid-gs/${dataset}/test/ann --bert_model bert-base-uncased --do_lower_case --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --output-conll preds.conll.${dataset}_uncased;
done;

### CASED
for dataset in i2b2_2006 physionet_goldstandard dernoncourt_goldstandard;
do python apply_model.py --model_dir models/${dataset} --input ../deid-gs/${dataset}/test/txt --labels ../deid-gs/${dataset}/test/ann --bert_model bert-base-cased --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --output-conll preds.conll.${dataset}_cased;
done

### LARGE
for dataset in i2b2_2006 i2b2_2014 physionet_goldstandard dernoncourt_goldstandard;
do python apply_model.py --model_dir models/${dataset}_large --input ../deid-gs/${dataset}/test/txt --labels ../deid-gs/${dataset}/test/ann --bert_model bert-large-cased --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --output-conll preds.conll.${dataset}_cased_large;
done;

### UNCASED LARGE
for dataset in i2b2_2006 i2b2_2014 physionet_goldstandard dernoncourt_goldstandard;
do python apply_model.py --model_dir models/${dataset}_uncased_large --input ../deid-gs/${dataset}/test/txt --labels ../deid-gs/${dataset}/test/ann --bert_model bert-large-uncased --do_lower_case --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --output-conll preds.conll.${dataset}_uncased_large;
done;


### CROSS-EVALUATION USING i2b2_2014_uncased (**small**)

base_model=i2b2_2014_uncased
for dataset in i2b2_2006 i2b2_2014 dernoncourt_goldstandard;
do CUDA_VISIBLE_DEVICES=1 python apply_model.py --model_dir models/${base_model} --input ../deid-gs/${dataset}/test/txt --labels ../deid-gs/${dataset}/test/ann --bert_model bert-base-uncased --do_lower_case --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --output-conll preds-cross-compare/preds.crosscompare.${base_model}.${dataset}_uncased;
done;


### CROSS EVAL ON RAD REPORT

base_model=i2b2_2014_uncased;
python apply_model.py --model_dir models/${base_model} --input ../deid-gs/rr-set-1/txt --labels ../deid-gs/rr-set-1/ann --bert_model bert-base-uncased --do_lower_case --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --output-conll preds.${base_model}.rr-set-1;
python apply_model.py --model_dir models/${base_model} --input ../deid-gs/rr-set-2/txt --labels ../deid-gs/rr-set-2/ann --bert_model bert-base-uncased --do_lower_case --task_name i2b2 -m overlap --step-size 100 --sequence-length 100 --output-conll preds.${base_model}.rr-set-2;

### RAD REPORTS (titanx)

python create_csv.py -i ../deid-gs/rr-set-1 -o data/rr/train.csv --bert_model bert-base-uncased -g -m overlap --sequence-length 100 --step-size 20
python create_csv.py -i ../deid-gs/rr-set-2 -o data/rr/test.csv --bert_model bert-base-uncased -g -m overlap --sequence-length 100 --step-size 20

## i2b2

First create a CSV dataset from XML:

```
python reformat-i2b2-to-deid.py -p i2b2/train -o i2b2/train-reformatted
python reformat-i2b2-to-deid.py -p i2b2/test-with-tags -o i2b2/test-reformatted

mv i2b2/test-reformatted/sentences.csv i2b2/dev.csv
mv i2b2/train-reformatted/sentences.csv i2b2/train.csv
``` 

`python bert_ner.py --data_dir i2b2data --bert_model bert-base-cased --task_name i2b2 --output_dir i2b2_model --do_train`

Convert output predictions to XML:

`python bert-pred-to-i2b2.py -p i2b2_model/eval_predictions.csv -d i2b2/test --bert_model bert-base-cased`

Reorganize into a single folder (run from bert-deid folder):

```python
import pandas as pd
import os
import shutil

df = pd.read_csv('debug_truncated_sentences.csv')

test_path = 'i2b2/test-with-tags'
model_pred_xml_path = 'output'
output_path = 'i2b2-eval-preds'
# get original files
files = os.listdir(test_path)
files = [f[0:-4] for f in files if f.endswith('.xml')]
N = len(files)
print(f'{N} total files in test.')
bad_docs = set(df['document_id'])

files = [f for f in files if f not in bad_docs]
N = len(files)
print(f'{N} files after removing docs with truncated sequences.')

os.mkdir(output_path)
os.mkdir(f'{output_path}/gold')
os.mkdir(f'{output_path}/pred')
for f in files:
    shutil.copyfile(f'{test_path}/{f}.xml', f'{output_path}/gold/{f}.xml')
    shutil.copyfile(f'{model_pred_xml_path}/{f}.xml', f'{output_path}/pred/{f}.xml')

```