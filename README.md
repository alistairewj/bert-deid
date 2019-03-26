# bert-deid

Code to fine-tune BERT on a medical note de-identification task.


CoNLL:

`python bert-ner.py --data_dir conll --bert_model bert-base-cased --task_name conll --output_dir conll_model --do_train --no_cuda --finetune`

i2b2:

`python bert-ner.py --data_dir i2b2data --bert_model bert-base-cased --task_name i2b2 --output_dir i2b2_model --do_train`