from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
import itertools
from bert_deid.ensemble_feature import find_phi_location, create_extra_feature_vector

import numpy as np
from tokenizers import (
    ByteLevelBPETokenizer, 
    SentencePieceBPETokenizer, 
    BertWordPieceTokenizer)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """Features are directly input to the transformer model."""
    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        input_subwords=None,
        input_offsets=None,
        input_lengths=None,
        extra_feature=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

        # offsets are used in evaluation to match predictions with text
        self.input_subwords = input_subwords
        self.input_offsets = input_offsets
        self.input_lengths = input_lengths

        self.extra_feature = extra_feature


def tokenize_with_labels(tokenizer, example, label_to_id, pad_token_label_id=-100, default_label='O'):
    ## TODO: update code for all tokenizer without fast
    text = example.text 
    labels = example.labels 

    # determine the type of tokenizer
    tokenizer_type = tokenizer.__class__.__name__
    if tokenizer_type == 'RobertaTokenizerFast':
        text = text.replace("\n", " ")


    encoding = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
    ex_ids, ex_offsets = encoding['input_ids'], encoding['offset_mapping']
    ex_tokens = tokenizer.tokenize(text)
    
    # get subword vectors
    ex_subwords = []
    for token in ex_tokens:
        if token.startswith('##') and tokenizer_type in ('BertTokenizerFast', 'DistilBertTokenizerFast'):
            # bert/distilbert add two hashes before sub-words
            ex_subwords.append(True)
        elif not token.startswith(b'\xe2\x96\x81'.decode('utf-8')) and \
        tokenizer_type in ('XLMRobertaTokenizer', 'CamembertTokenizer', 'AlbertTokenizer'):
            # more recent tokenizers add a special chars where a whitespace prefixed a word
            # e.g. big tokenizer -> ['_big', '_to', 'ken', 'izer] where '_' == \xe2\x96\x81
            ex_subwords.append(True)
        elif tokenizer_type in ('RobertaTokenizerFast'):
            # add Ġ where a whitespace prefixed a word 
            if token != b'\xc4\xa0'.decode('utf-8') and not token.startswith(b'\xc4\xa0'.decode('utf-8')):
                ex_subwords.append(True)
            else:
                ex_subwords.append(False)
        else:
            ex_subwords.append(False)
            
    # get token labels
    ex_labels = [
            pad_token_label_id if sw else default_label for sw in ex_subwords
        ]

    ex_label_ids = [
        label_to_id[l] if l != pad_token_label_id else pad_token_label_id
        for l in ex_labels]

    # when building examples for model evaluation, there are no labels
    if labels is None:
        return ex_ids, ex_tokens, ex_offsets, ex_subwords, ex_label_ids

    labels = sorted(labels, key=lambda label: label.start)

    i = 0
    l = 0
    while i < len(ex_tokens) and l < len(labels):
        label = labels[l]
        entity_type = label.entity_type.upper()
        start, stop = label.start, label.start + label.length
        start_i, stop_i = ex_offsets[i]
        if start_i >= start and stop_i <= stop and not ex_subwords[i]:
            ex_labels[i] = entity_type
        if start_i > stop:
            l += 1
        else:
            i += 1

    ex_label_ids = [
        label_to_id[l] if l != pad_token_label_id else pad_token_label_id
        for l in ex_labels]

    assert len(ex_ids) == len(ex_tokens)
    assert len(ex_offsets) == len(ex_tokens)
    assert len(ex_subwords) == len(ex_tokens)
    assert len(ex_label_ids) == len(ex_tokens)
            
    return ex_ids, ex_tokens, ex_offsets, ex_subwords, ex_label_ids

def convert_examples_to_features(
    examples,
    label_to_id,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token="[PAD]",
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    feature_overlap=None,
):

    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

        `feature_overlap` - Split a single long example into multiple training observations. This is
        useful for handling examples containing very long passages of text.
            None (default): truncates each example at max_seq_length -> one InputFeature per InputExample.
            [0, 1): controls how much overlap between consecutive segments.
                e.g. `feature_overlap=0.1` means the last 10% of InputFeature 1 will equal first 10%
                of InputFeature 2, assuming that the InputExample is long enough to require splitting.
    """

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2

    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    cls_token_id = tokenizer.convert_tokens_to_ids(cls_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    features = []
    n_obs = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        patterns = example.patterns
        pattern_label = 1
        ex_phi_locs = []
        for pattern in patterns:
            ex_phi_locs.append(find_phi_location(pattern, pattern_label, example.text))

        assert (len(patterns) == len(ex_phi_locs))

        ex_ids, ex_tokens, ex_offsets, ex_subwords, ex_label_ids = tokenize_with_labels(
            tokenizer, example, label_to_id, pad_token_label_id=pad_token_label_id
        )

        n_tokens = len(ex_ids)
        feature_overlap = 0
        if feature_overlap is None:
            # we will truncate the sequence by having the iterator only have start=0
            feature_len = n_tokens
        else:
            feature_len = int(
                (1 - feature_overlap) * (max_seq_length - special_tokens_count)
            )

        token_iterator = range(0, n_tokens, feature_len)
        for t in token_iterator:
            input_ids = ex_ids[t:t + max_seq_length - special_tokens_count]
            tokens = ex_tokens[t:t + max_seq_length - special_tokens_count]
            offsets = ex_offsets[t:t + max_seq_length - special_tokens_count]
            subwords = ex_subwords[t:t + max_seq_length - special_tokens_count]
            label_ids = ex_label_ids[t:t + max_seq_length - special_tokens_count]

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

            input_ids += [sep_token_id]
            tokens += [sep_token]
            offsets += [(-1, -1)]
            subwords += [False]
            label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            
            if cls_token_at_end:
                # for XLNet [cls] at the end
                input_ids += [cls_token_id]
                tokens += [cls_token]
                offsets += [(-1, -1)]
                subwords += [False]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                input_ids = [cls_token_id] + input_ids
                tokens = [cls_token] + tokens
                offsets = [(-1, -1)] + offsets
                subwords = [False] + subwords
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                
            
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token_id] * padding_length) + input_ids
                input_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = (
                    [pad_token_segment_id] * padding_length
                ) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                offsets = ([(-1, -1)] * padding_length) + offsets
                subwords = ([False] * padding_length) + subwords
            else:
                input_ids += [pad_token_id] * padding_length
                input_mask += [
                    0 if mask_padding_with_zero else 1
                ] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                offsets += [(-1, -1)] * padding_length
                subwords += [False] * padding_length

            extra_features = []
            for i in range(len(ex_phi_locs)):
                extra_feature = create_extra_feature_vector(ex_phi_locs[i], offsets, lengths, token_sw)
                extra_features.append(extra_feature)

            starts = [start for start, _ in offsets]
            lengths = [stop-start for start, stop in offsets]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(starts) == max_seq_length
            assert len(lengths) == max_seq_length
            assert len(subwords) == max_seq_length
            if len(extra_features) > 0:
                assert len(extra_features[0]) == max_seq_length
                assert len(extra_features) == len(example.patterns)

            if n_obs < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info(
                    "subwords: %s", " ".join([str(x)[0] for x in subwords])
                )
                logger.info(
                    "input_ids: %s", " ".join([str(x) for x in input_ids])
                )
                logger.info(
                    "input_mask: %s", " ".join([str(x) for x in input_mask])
                )
                logger.info(
                    "segment_ids: %s", " ".join([str(x) for x in segment_ids])
                )
                logger.info(
                    "label_ids: %s", " ".join([str(x) for x in label_ids])
                )
                logger.info("offsets: %s", " ".join([str(x) for x in starts]))
                for each_feature in extra_features:
                    logger.info('extra feature: %s', ' '.join([str(x) for x in each_feature]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                    input_offsets=starts,
                    input_lengths=lengths,
                    input_subwords=subwords,
                    extra_feature=extra_features,
                )
            )
            n_obs += 1
    return features


        



