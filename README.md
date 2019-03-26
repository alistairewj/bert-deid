# bert-deid

Code to fine-tune BERT on a medical note de-identification task.


CoNLL:

`python bert_ner.py --data_dir conll --bert_model bert-base-cased --task_name conll --output_dir conll_model --do_train --do_eval`

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