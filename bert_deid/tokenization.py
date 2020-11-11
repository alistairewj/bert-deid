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

from typing import List, Tuple
import collections
import logging
import os
import re
import unicodedata
import itertools
from bisect import bisect_left, bisect_right

from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd
import tokenizers

from bert_deid.processors import InputFeatures

logger = logging.getLogger(__name__)

philter_pred = '/home/jingglin/research/philter/all/test/pred/'


def pattern_spans(text, pattern):
    """
    Iterator that splits text using a regex pattern.
    
    Returns
    -------
    token, start, stop
        Tuple containing the token, the start index of the token
        in the original string, and the end index of the
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


def print_tokens_with_text(text, tokens, offsets, lengths):
    for i, token in enumerate(tokens):
        start, stop = offsets[i], offsets[i] + lengths[i]
        print('{:10s} {:10s}'.format(token, text[start:stop]))


def get_token_labels(
    tokenizer, encoded, example, pad_token_label_id=-100, default_label='O'
):
    # construct sub-words flags
    # TODO: does this vary according to model?
    token_sw = [False] + [
        encoded.words[i + 1] == encoded.words[i]
        for i in range(len(encoded.words) - 1)
    ]

    # initialize token labels as the default label
    # set subword tokens to padded token
    token_labels = [
        pad_token_label_id if sw else default_label for sw in token_sw
    ]

    # when building examples for model evaluation, there are no labels
    if example.labels is None:
        return token_labels

    offsets = [o[0] for o in encoded.offsets]
    lengths = [o[1] - o[0] for o in encoded.offsets]
    w = 0
    for label in example.labels:
        entity_type = label.entity_type
        start, offset = label.start, label.length
        stop = start + offset

        # get the first offset > than the label start index
        i = bisect_left(offsets, start)
        if i == len(offsets):
            # we have labeled a token at the end of the text
            # also catches the case that we label a part of a token
            # at the end of the text, but not the entire token
            if not token_sw[-1]:
                token_labels[-1] = entity_type
        else:
            # find the last token which is within this label
            j = bisect_left(offsets, stop)

            # assign all tokens between [start, stop] to this label
            # *except* if it is a padding token (so the model ignores subwords)
            new_labels = [
                entity_type if not token_sw[k] else pad_token_label_id
                for k in range(i, j)
            ]
            token_labels[i:j] = new_labels
    return token_labels


def tokenize_with_labels(
    tokenizer, example, pad_token_label_id=-100, default_label='O'
):
    text = example.text

    # tokenize the text, retain offsets, subword locations, and lengths
    encoded = tokenizer.encode(text)
    offsets = [o[0] for o in encoded.offsets]
    lengths = [o[1] - o[0] for o in encoded.offsets]

    # TODO: do we need to fix it?
    # fix the offset of the final token, if special
    # if offsets[-1] == 0:
    #     offsets[-1] = len(text)

    word_tokens = encoded.tokens
    # construct sub-words flags
    # TODO: does this vary according to model?
    token_sw = [False] + [
        encoded.words[i + 1] == encoded.words[i]
        for i in range(len(encoded.words) - 1)
    ]

    token_labels = get_token_labels(
        tokenizer, encoded, example, pad_token_label_id=-100, default_label='O'
    )

    return word_tokens, token_labels, token_sw, offsets, lengths


def map_tags_to_tokens(example, offsets):
    # initialize feature as 0
    # set subword tokens to padded token
    tags = collections.OrderedDict()

    for tag in example.tags:
        name = tag.name
        start, offset = tag.start, tag.length
        stop = start + offset

        if name not in tags:
            # initialize feature to all 0s
            tags[name] = np.zeros(len(offsets), dtype=int)

        # get the first offset > than the label start index
        i = bisect_left(offsets, start)
        if i == len(offsets):
            # we have labeled a token at the end of the text
            # also catches the case that we label a part of a token
            # at the end of the text, but not the entire token
            tags[name][-1] = 1
        else:
            # find the last token which is within this label
            j = bisect_left(offsets, stop)

            # assign all tokens between [start, stop] to this label
            tags[name][i:j] = 1

    # convert tags to an NxM numpy array
    # N is the number of tokens, M is the number of distinct entities
    tags = np.column_stack(list(tags.values()))
    return tags


def align_predictions(
    predictions: np.ndarray, label_ids: np.ndarray, label_map: dict
) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size = preds.shape[0]

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    # ignore instances where label_id == cross entropy ignore
    idx = label_ids != CrossEntropyLoss().ignore_index
    out_label_list = []
    preds_list = []
    for i in range(batch_size):
        out_label_list.append(list(map(label_map.get, label_ids[i, idx[i, :]])))
        preds_list.append(list(map(label_map.get, preds[i, idx[i, :]])))

    return preds_list, out_label_list
