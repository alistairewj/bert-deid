"""Class for applying BERT-deid on text."""
import os
import re
import logging
from hashlib import sha256
from dataclasses import astuple, dataclass, fields
from typing import List, Optional, Union, TextIO

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)

from bert_deid.processors import InputFeatures, TokenClassificationTask, Split

# custom class written for albert token classification
from bert_deid import tokenization, processors
from bert_deid.label import LabelCollection, LABEL_SETS, LABEL_MEMBERSHIP
from bert_deid.processors import InputExample

logger = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    input_subwords: Optional[List[int]] = None
    offsets: Optional[List[int]] = None
    lengths: Optional[List[int]] = None


class Transformer(object):
    """Wrapper for a Transformer model to be applied for NER."""
    def __init__(
        self,
        model_path,
        # token_step_size=100,
        # sequence_length=100,
        max_seq_length=128,
        device='cpu',
    ):
        # by default, we do non-overlapping segments of text
        # self.token_step_size = token_step_size
        # sequence_length is how long each example for the model is
        # self.sequence_length = sequence_length

        # get the definition classes for the model
        # training_args = torch.load(
        #     os.path.join(model_path, 'training_args.bin')
        # )

        # task applied
        # TODO: figure out how to load this from saved model
        label_set = LabelCollection('i2b2_2014', transform='simple')
        self.token_classification_task = processors.DeidProcessor(
            data_dir='', label_set=label_set
        )

        # Load pretrained model and tokenizer
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            # we always use fast tokenizers as we need offsets from tokenization
            use_fast=True,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            from_tf=False,
            config=self.config,
        )

        # max seq length is what we pad the model to
        # max seq length should always be >= sequence_length + 2
        self.max_seq_length = self.config.max_position_embeddings

        label_map = self.config.id2label
        self.labels = [label_map[i] for i in range(len(label_map))]
        self.num_labels = len(self.labels)

        # Use cross entropy ignore index as padding label id so
        # that only real label ids contribute to the loss later
        # TODO: get this from the model
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        # prepare the model for evaluation
        # CPU probably faster, avoids overhead
        self.device = torch.device(device)
        self.model.to(self.device)

    def split_by_overlap(self, text, token_step_size=20, sequence_length=100):
        # track offsets in tokenization
        tokens, tokens_sw, tokens_idx = self.tokenizer.tokenize_with_index(text)

        if len(tokens_idx) == 0:
            # no tokens found, return empty list
            return []
        # get start index of each token
        tokens_start = [x[0] for x in tokens_idx]
        tokens_start = np.array(tokens_start)

        # forward fill index for first token over its subsequent subword tokens
        # this means that if we try to split on a subword token, we will actually
        # split on the starting word
        mask = np.array(tokens_sw) == 1
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        tokens_start[mask] = tokens_start[idx[mask]]

        if len(tokens) <= sequence_length:
            # very short text - only create one example
            seq_offsets = [[tokens_start[0], len(text)]]
        else:
            seq_offsets = range(
                0,
                len(tokens) - sequence_length, token_step_size
            )
            last_offset = seq_offsets[-1] + token_step_size
            seq_offsets = [
                [tokens_start[x], tokens_start[x + sequence_length]]
                for x in seq_offsets
            ]

            # last example always goes to the end of the text
            seq_offsets.append([tokens_start[last_offset], len(text)])

        # turn our offsets into examples
        # create a list of lists, each sub-list has 4 elements:
        #   sentence number, start index, end index, text of the sentence
        examples = list()

        for i, (start, stop) in enumerate(seq_offsets):
            examples.append([i, start, stop, text[start:stop]])

        return examples

    def _split_text_into_segments(self, text, feature_overlap=None):
        """Splits text into overlapping segments based on the model sequence length."""
        # tokenize the example text
        encoded = self.tokenizer._tokenizer.encode(
            text, add_special_tokens=False
        )
        token_sw = [False] + [
            encoded.words[i + 1] == encoded.words[i]
            for i in range(len(encoded.words) - 1)
        ]
        token_offsets = np.array(encoded.offsets)

        seq_len = self.tokenizer.max_len_single_sentence
        if feature_overlap is None:
            feature_overlap = 0
        # identify the starting offsets for each sub-sequence
        new_seq_jump = int((1 - feature_overlap) * seq_len)

        # iterate through subsequences and add to examples
        inputs = []
        start = 0
        while start < token_offsets.shape[0]:
            # ensure we do not start on a sub-word token
            while token_sw[start]:
                start -= 1

            stop = start + seq_len
            if stop < token_offsets.shape[0]:
                # ensure we don't split sequences on a sub-word token
                # do this by shortening the current sequence
                while token_sw[stop]:
                    stop -= 1
            else:
                # end the sub sequence at the end of the text
                stop = token_offsets.shape[0] - 1

            text_subseq = text[token_offsets[start, 0]:token_offsets[stop, 0]]
            encoded = self.tokenizer._tokenizer.encode(text_subseq)
            encoded.pad(self.tokenizer.model_max_length)

            subseq_sw = [False] + [
                encoded.words[i + 1] == encoded.words[i]
                for i in range(len(encoded.words) - 1)
            ]

            inputs.append(
                InputFeatures(
                    input_ids=encoded.ids,
                    attention_mask=encoded.attention_mask,
                    token_type_ids=encoded.type_ids,
                    input_subwords=subseq_sw,
                    # note the offsets are based off the original text, not the subseq
                    offsets=[
                        o[0] + token_offsets[start, 0] for o in encoded.offsets
                    ],
                    lengths=[o[1] - o[0] for o in encoded.offsets]
                )
            )

            # update start of next sequence to be end of current one
            start = start + new_seq_jump

        return inputs

    def _features_to_tensor(self, inputs):
        """Extracts tensor datasets from a list of InputFeatures"""

        # create tensor datasets for model input
        input_ids = torch.tensor(
            [x.input_ids for x in inputs], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [x.attention_mask for x in inputs], dtype=torch.long
        )
        token_type_ids = torch.tensor(
            [x.token_type_ids for x in inputs], dtype=torch.long
        )
        return input_ids, attention_mask, token_type_ids

    def _logits_to_standoff(self, logits, inputs, ignore_label='O'):
        """Converts prediction logits to stand-off prediction labels."""
        # mask, offsets, lengths
        # convert logits to probabilities

        # extract most likely label for each token
        # prob is used to decide between overlapping labels later
        pred_id = np.argmax(logits, axis=2)
        # calculate softmax
        probs = np.exp(logits - np.expand_dims(np.max(logits, axis=2), 2))
        probs = probs / np.expand_dims(probs.sum(axis=2), 2)
        # get max probability
        probs = np.max(probs, axis=2)

        # re-align the predictions with the original text
        # across each sub sequence..
        labels = []
        for i in range(logits.shape[0]):
            # extract mask for valid tokens, offsets, and lengths
            mask = np.asarray(inputs[i].attention_mask).astype(bool)
            offsets, lengths = inputs[i].offsets, inputs[i].lengths
            subwords = inputs[i].input_subwords

            # increment the lengths for the first token in words tokenized into subwords
            # this ensures a prediction covers the subsequent subwords
            # it assumes that predictions are made for the first token in sub-word tokens
            # TODO: assert the model only predicts a label for first sub-word token
            lengths = inputs[i].lengths
            for j in reversed(range(len(subwords))):
                if subwords[j]:
                    # cumulatively sums lengths for subwords until the first subword token
                    lengths[j - 1] += lengths[j]

            pred_label = [self.config.id2label[p] for p in pred_id[i, :]]
            # ignore object labels
            mask = mask & ~(
                np.asarray([p == ignore_label
                            for p in pred_label]).astype(bool)
            )

            # ignore CLS and SEP tokens
            mask = mask & np.asarray(
                [
                    (inp != self.tokenizer.cls_token_id) &
                    (inp != self.tokenizer.sep_token_id)
                    for inp in inputs[i].input_ids
                ]
            ).astype(bool)

            # keep (1) unmasked labels where (2) it is not an object (the default category)
            # and (3) it is not a special CLS/SEP token
            idxKeep = np.where(mask)[0]

            labels.extend(
                [
                    [probs[i, p], pred_label[p], offsets[p], lengths[p]]
                    for p in idxKeep
                ]
            )

        # now, we may have multiple predictions for the same offset token
        # this can happen as we are overlapping observations to maximize
        # context for tokens near the window edges
        # so we take the *last* prediction, because that prediction will have
        # the most context

        # np.unique returns index of first unique value, so reverse the list
        offsets = [l[2] for l in labels]
        offsets.reverse()
        _, unique_idx = np.unique(offsets, return_index=True)
        unique_idx = len(offsets) - unique_idx - 1
        labels = [labels[i] for i in unique_idx]

        return labels

    def predict(self, text, batch_size=8, num_workers=0, feature_overlap=None):
        # sets the model to evaluation mode to fix parameters
        self.model.eval()

        # create a dictionary with inputs to the model
        # each element is a list of the sequence data
        inputs = self._split_text_into_segments(text, feature_overlap)
        input_ids, attention_mask, token_type_ids = self._features_to_tensor(
            inputs
        )

        # ensure input is on same device as model
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        token_type_ids = token_type_ids.to(self.model.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )[0]

        logits = logits.detach().cpu().numpy()
        preds = self._logits_to_standoff(logits, inputs, ignore_label='O')

        # returns a list of the predictions with the token span
        return preds

    def apply(self, text, repl='___'):
        preds = self.predict(text)
        if len(preds) == 0:
            return text

        extend_token = 0
        for i in reversed(range(len(preds))):
            start, length = preds[i][2:]
            # for consecutive entities, only insert a single replacement
            if i > 0:
                if (preds[i - 1][2] + preds[i - 1][3]) == start:
                    extend_token += length
                    continue

            # replace this instance of text with three underscores
            text = text[:start] + repl + text[start + length + extend_token:]
            extend_token = 0

        return text
