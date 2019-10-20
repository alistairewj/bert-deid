# bert-deid

Code to fine-tune BERT on a medical note de-identification task.

Install:

```
python setup.py sdist; python -m pip install -U dist/bert_deid-0.1.tar.gz
```

## Use wrappers to generate data and train model

The following guide assumes you are running the code on quadro.

prepare the train.csv/test.csv files:

`python scripts/create_csv_config.py -c configs/quadro/token/i2b2_2014.json`

train the model (`CUDA_VISIBLE_DEVICES=0 ` is used to ignore the other, outdated GPU on the machine):

`CUDA_VISIBLE_DEVICES=0 python bert_deid/train_bert.py --data_path /home/alistairewj/git/bert-deid/data/token/i2b2_2014 --bert_model bert-base-uncased --task_name i2b2 --model_path /home/alistairewj/git/bert-deid/models/token/i2b2_2014 --max_seq_length 128 --do_lower_case --seed 800656 --do_train`

create a list of all tokens in the training set:

`python scripts/create_train_tokens_file.py -c configs/quadro/token/i2b2_2014.json`

make predictions on a test set and evaluate the model:

`python scripts/run_and_eval_bert.py -c configs/quadro/token/i2b2_2014.json`

calculate operating point statistics on that test set:

`python scripts/eval_pred.py  -c configs/quadro/token/i2b2_2014.json`
