# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright 2019 Alistair Johnson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
import itertools
from bisect import bisect_left

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """Features are directly input to the transformer model."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def pattern_spans(text, pattern):
    """
    Iterator that splits text using a regex pattern.
    
    Returns
    -------
    token, start, offset
        Tuple containing the token, the start index of the token
        in the original string, and the length of the token ("offset")
    """

    tokens = pattern.split(text)

    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)


def split_by_pattern(text, pattern):
    """
    Function that wraps the pattern span iterator.
    """
    tokens_with_spans = list()
    n = 0
    for token, start, end in pattern_spans(text, pattern):
        tokens_with_spans.append([n, start, end, token])
        n += 1

    return tokens_with_spans


def tokenize_with_labels(
    tokenizer, example, pad_token_label_id=-100, default_label='O'
):
    text = example.text
    if '##' in text:
        logger.warning('Text contains hashes. Offset code may break.')
        i = text.index("##")
        start, stop = min(i - 10, i), max(i + 10, len(text))
        logger.warning(f'bad text: {text[start:stop]}')

    word_tokens = tokenizer.tokenize(text)

    if hasattr(tokenizer, 'basic_tokenizer'):
        if hasattr(tokenizer.basic_tokenizer, 'do_lower_case'):
            if tokenizer.basic_tokenizer.do_lower_case:
                text = text.lower()

    # now reverse engineer the locations of the tokens
    offsets = []
    w = 0
    i = 0

    while i < len(text):
        if w >= len(word_tokens):
            # we have assigned all word tokens
            # what remains must be whitespace/filtered characters
            break

        token = word_tokens[w]

        if token == '[UNK]':
            # can't do this token
            raise ValueError(
                'Unable to get offsets for tokens due to unknown tokens.'
            )

        # ignore the ##s added by the tokenizer
        if token.startswith('##'):
            token = token[2:]

        # end if we have reached the end of the text
        if (i + len(token)) > len(text):
            break

        # if the token is identical to the text, store the offset
        if text[i:i + len(token)] == token:
            offsets.append(i)
            w += 1
            i += len(token)
            continue

        i += 1

    # initialize token labels as the default label
    token_labels = [default_label] * len(word_tokens)
    w = 0
    for label, start, offset in example.labels:
        stop = start + offset
        # HACK: force all labels to be upper case
        label = label.upper()
        # get the first offset > than the label start index
        i = bisect_left(offsets, start)
        if i == len(offsets):
            # we have labeled a token at the end of the text
            # also catches the case that we label a part of a token
            # at the end of the text, but not the entire token
            token_labels[-1] = label
        else:
            # assign all tokens between [start, stop] to this label
            j = bisect_left(offsets, stop)
            token_labels[i:j] = [label] * (j - i)

    return word_tokens, offsets, token_labels


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens, offsets, labels = tokenize_with_labels(
            tokenizer, example, pad_token_label_id=pad_token_label_id
        )

        # assign labels based off the offsets
        label_ids = [
            label_map[l] if l != pad_token_label_id else pad_token_label_id
            for l in labels
        ]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            offsets = offsets[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        offsets += [-1]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            offsets += [-1]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            offsets += [-1]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            offsets = [-1] + offsets
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = (
                [pad_token_segment_id] * padding_length
            ) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s", " ".join([str(x) for x in input_mask])
            )
            logger.info(
                "segment_ids: %s", " ".join([str(x) for x in segment_ids])
            )
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids
            )
        )
    return features
