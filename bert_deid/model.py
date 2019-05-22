"""Class for applying BERT-deid on text."""
import os
import re

import numpy as np
import pandas as pd

import torch
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pydeid import annotation

from bert_deid.bert_ner import BertForNER
from bert_deid.bert_multilabel import BertMultiLabel
from bert_deid.create_csv import split_by_overlap
from bert_deid.tokenization import BertTokenizerNER


class BertForDEID(BertForNER):
    """BERT model for deidentification."""

    def __init__(self, model_dir,
                 max_seq_length=100,
                 token_step_size=100):
        self.labels = [
            'NAME', 'LOCATION', 'AGE',
            'DATE', 'ID', 'CONTACT', 'O',
            'PROFESSION'
        ]
        self.num_labels = len(self.labels)
        self.label_id_map = {i: label for i, label in enumerate(self.labels)}
        self.bert_model = 'bert-base-uncased'
        self.do_lower_case = True

        # by default, we do non-overlapping segments of text
        self.max_seq_length = max_seq_length
        self.token_step_size = token_step_size

        # load trained config/weights
        model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config_file = os.path.join(model_dir, CONFIG_NAME)
        if os.path.exists(model_file) & os.path.exists(config_file):
            print(f'Loading model and configuration from {model_dir}.')
            config = BertConfig(config_file)
            super(BertForDEID, self).__init__(config, self.num_labels)
            self.load_state_dict(torch.load(model_file, map_location="cpu"))
        else:
            raise ValueError('Folder %s did not have model and config file.',
                             model_dir)

        # initialize tokenizer
        self.tokenizer = BertTokenizerNER.from_pretrained(
            self.bert_model, do_lower_case=self.do_lower_case)

        # prepare the model for evaluation
        # CPU probably faster, avoids overhead
        device = torch.device("cpu")
        self.to(device)

        # for post-fixes

        # lower case obvious false positives
        self.fp_words = set(['swan-ganz', 'swan ganz',
                             'carina', 'dobbhoff', 'shiley',
                             'hcc', 'kerley', 'technologist'])

    def _prepare_tokens(self, tokens, tokens_sw, tokens_idx):
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return input_ids, input_mask, segment_ids

    def annotate(self, text, annotations=None, document_id=None, column=None,
                 **kwargs):
        # annotate a string of text

        # split the text into examples
        # we choose non-overlapping examples for evaluation
        examples = split_by_overlap(
            text, self.tokenizer,
            token_step_size=self.token_step_size,
            max_seq_len=self.max_seq_length
        )

        anns = list()

        # examples is a list of lists, each sub-list has 4 elements:
        #   sentence number, start index, end index, text of the sentence
        for e, example in enumerate(examples):
            # track offsets in tokenization
            tokens, tokens_sw, tokens_idx = self.tokenizer.tokenize(
                example[3])

            # offset index of predictions based upon example start
            tokens_offset = example[1]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens_sw = [0] + tokens_sw + [0]
            tokens_idx = [[-1]] + tokens_idx + [[-1]]

            input_ids, input_mask, segment_ids = self._prepare_tokens(
                tokens, tokens_sw, tokens_idx
            )

            # convert to tensor
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)

            # reshape to [1, SEQ_LEN] as this is expected by model
            input_ids = input_ids.view([1, input_ids.size(0)])
            input_mask = input_mask.view([1, input_mask.size(0)])
            segment_ids = segment_ids.view([1, segment_ids.size(0)])

            # make predictions
            with torch.no_grad():
                logits = self.forward(input_ids, segment_ids, input_mask)

            # convert to list // get the first element as we have only 1 example
            scores = logits.tolist()[0]
            pred = np.argmax(logits, axis=-1).tolist()[0]

            # remove [CLS] at beginning
            tokens = tokens[1:]
            tokens_sw = tokens_sw[1:]
            tokens_idx = tokens_idx[1:]
            pred = pred[1:]
            scores = scores[1:]

            # subselect pred and mask down to length of tokens
            last_token = tokens.index('[SEP]')
            tokens = tokens[:last_token]
            tokens_sw = tokens_sw[:last_token]
            tokens_idx = tokens_idx[:last_token]
            pred = pred[:last_token]
            scores = scores[:last_token]

            if len(tokens) == 0:
                # no data for predictions
                continue

            # the model does not make predictions for sub-words
            # the first word-part for a segmented sub-word is used as the prediction
            # so we append sub-word indices to previous non-subword indices
            k = 0
            while k < len(tokens_sw):
                if tokens_sw[k] == 1:
                    # pop these indices
                    current_idx = tokens_idx.pop(k)
                    # add sub-word index to previous non-subword
                    tokens_idx[k-1].extend(current_idx)

                    token_add = tokens.pop(k)
                    tokens[k-1] = tokens[k-1] + token_add[2:]
                    # remove the token from other lists
                    scores.pop(k)
                    pred.pop(k)
                    tokens_sw.pop(k)
                else:
                    k += 1

            for k in range(len(pred)):

                # note we offset it to the document using tokens_offset
                start = tokens_idx[k][0] + tokens_offset
                stop = tokens_idx[k][-1] + tokens_offset + 1

                if self.label_id_map[pred[k]] == 'O':
                    continue

                # if we are here, we have a prediction
                # add this to the output
                row = [
                    # document_id,annotation_id,annotator
                    document_id, f'bert.{e}.{k}', self.bert_model,
                    # start,stop
                    start, stop,
                    # entity
                    tokens[k],
                    # entity_type (prediction)
                    self.label_id_map[pred[k]],
                    # comment
                    None,
                    # confidence - the score assigned by bert
                    scores[k][pred[k]]
                ]
                anns.append(row)

        df = pd.DataFrame(anns, columns=['document_id', 'annotation_id',
                                         'annotator', 'start', 'stop',
                                         'entity', 'entity_type',
                                         'comment', 'confidence'])
        df['start'] = df['start'].astype(int)
        df['stop'] = df['stop'].astype(int)

        return df

    def pool_annotations(self, df):
        # pool token-wise annotations together
        # this is necessary if overlapping examples are used
        # i.e. self.token_step_size < self.sequence_length

        # get location of maximally confident annotations
        df_keep = df.groupby(['annotator', 'start', 'stop'])[
            ['confidence']].max()
        df_keep.reset_index(inplace=True)

        # merge on these columns to remove rows with low confidence
        grp_cols = list(df_keep.columns)
        df = df.merge(df_keep, how='inner', on=grp_cols)

        # if two rows had identical confidence, keep the first
        df.drop_duplicates(subset=grp_cols, keep='first', inplace=True)

        return df

    def postfix(self, df, text):
        """Post-hoc corrections using rules.
        Designed using radiology reports."""
        idxKeep = list()
        for i, row in df.iterrows():
            if row['entity'].lower() in self.fp_words:
                continue

            if 'swan' in row['entity'].lower():
                if text[row['start']:row['start']+9].lower() in self.fp_words:
                    continue

            if row['entity_type'].lower() == 'age':
                # remove 'M' or 'F' from entities
                if (len(row['entity']) == 3) & (row['entity'][-1].lower() in ('m', 'f')):
                    df.loc[i, 'stop'] -= 1
                    df.loc[i, 'entity'] = df.loc[i, 'entity'][0:-1]
            elif row['entity_type'].lower() == 'date':
                # remove dates which are from bulleted lists
                if re.search(r'\n [0-9][.)] ', text[row['start']-2:row['stop']+2]):
                    continue
            elif row['entity_type'].lower() == 'location':
                if row['entity'].lower() in ['ge', 'hickman', 'carina']:
                    continue

            idxKeep.append(i)

        return df.loc[idxKeep, :]
