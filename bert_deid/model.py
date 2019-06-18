"""Class for applying BERT-deid on text."""
import os
import re

import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForTokenClassification

from bert_deid.create_csv import split_by_overlap
from bert_deid.tokenization import BertTokenizerNER
import bert_deid.processors as processors


def segment_ids(self, segment1_len, segment2_len):
    ids = [0] * segment1_len + [1] * segment2_len
    return torch.tensor([ids]).to(device=self.device)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def prepare_tokens(tokens, tokens_sw, tokens_idx, label,
                   label_list, max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # example.label is a list of start/stop offsets for tagged entities
    # use this to create list of labels for each token
    # assumes labels are ordered
    labels = ['O'] * len(input_ids)
    if len(label) > 0:
        l_idx = 0
        start, stop, entity = label[l_idx]
        for i, idx in enumerate(tokens_idx):
            if idx[0] >= stop:
                l_idx += 1
                # exit loop if we have finished assigning labels
                if l_idx >= len(label):
                    break
                start, stop, entity = label[l_idx]

            if (idx[0] >= start) & (idx[0] < stop):
                # tokens are assigned based on label of first character
                labels[i] = entity

    # convert from text labels to integer coding
    # if a label is not in the map, we ignore that token in backprop
    labels[0] = "[CLS]"
    labels[-1] = "[SEP]"
    label_ids = [label_map[x]
                 if x in label_map
                 else -1
                 for x in labels]

    # set labels for subwords to -1 to ignore them in label loss
    label_ids = [-1 if tokens_sw[i] == 1
                 else label_id
                 for i, label_id in enumerate(label_ids)]

    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    label_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, label_ids


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        # track offsets in tokenization
        tokens_a, tokens_a_sw, tokens_a_idx = tokenizer.tokenize_with_index(
            example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            tokens_a_sw = tokens_a_sw[:(max_seq_length - 2)]
            tokens_a_idx = tokens_a_idx[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens_sw = [0] + tokens_a_sw + [0]
        tokens_idx = [[-1]] + tokens_a_idx + [[-1]]

        input_ids, input_mask, segment_ids, label_ids = prepare_tokens(
            tokens, tokens_sw, tokens_idx, example.label,
            label_list, max_seq_length, tokenizer)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" %
        #                 " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" %
        #                 " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s" %
        #                 " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BertForNER(BertForTokenClassification):
    """BERT model for neural entity recognition.
    Essentially identical to BertForTokenClassification, but ignores
    labels with an index of -1 in the loss function.
    """

    def __init__(self, config, num_labels):
        super(BertForNER, self).__init__(config, num_labels)

    def forward(self, input_ids,
                token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForDEID(BertForNER):
    """BERT model for deidentification."""

    def __init__(self, model_dir,
                 max_seq_length=100,
                 token_step_size=100,
                 task_name='i2b2'):

        # use the associated data processor to define label set
        proc_dict = {
            "conll": processors.CoNLLProcessor,
            "hipaa": processors.hipaaDeidProcessor,
            "binary": processors.binaryDeidProcessor,
            "i2b2": processors.i2b2DeidProcessor
        }
        if task_name not in proc_dict:
            raise ValueError('Unrecognized task: %s', task_name)

        self.labels = proc_dict[task_name]().get_labels()
        self.num_labels = len(self.labels)
        self.label_id_map = {i: label for i, label in enumerate(self.labels)}

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

        if config.num_hidden_layers == 12:
            self.bert_model = 'bert-base'
        else:
            # num_hidden_layers == 24
            self.bert_model = 'bert-large'

        if config.vocab_size == 28996:
            self.bert_model += '-cased'
            self.do_lower_case = False
        else:
            # vocab_size == 30522
            self.bert_model += '-uncased'
            self.do_lower_case = True

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
            tokens, tokens_sw, tokens_idx = self.tokenizer.tokenize_with_index(
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
        if df.shape[0] == 0:
            return df

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
