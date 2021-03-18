import re
import csv
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from enum import Enum
import sys
import logging
from collections import namedtuple
from typing import List, Optional, Union, TextIO

import numpy as np

from bert_deid.label import Label, LabelCollection

logger = logging.getLogger(__name__)

Tag = namedtuple('Tag', ['name', 'start', 'offset'])


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    
    
    (OLD) Args:
        guid: Unique id for the example.
        text: string. The text comprising the sequence.
        label: (Optional) list. Stand-off labels. List of lists.
            Each element has the form:
            (label_name, start_offset, length)
        tags: (Optional) list. Stand-off features. List of lists.
            Additional features for e.g. tagged entities. Expected format is as
            a list of NamedTuples: [(name, start, length), ...].
            These are converted into binary features and concatenated to the input.
        
        Note: If tokenization results in a token partially tagged by a feature,
        the feature will be expanded to cover the entire token.
    """

    guid: str
    text: str
    labels: Optional[List[str]]
    tags: Optional[List] = None
    # self.text = text
    # self.tags = tags


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
    # input_subwords: Optional[List[int]] = None
    # input_offsets: Optional[List[int]] = None
    # input_lengths: Optional[List[int]] = None
    # additional_features: Optional[List] = None


class TokenClassificationTask(object):
    """Base class for data converters."""
    def __init__(self, data_dir):
        """Initialize a data processor with the location of the data."""
        self.data_dir = data_dir
        self.data_filenames = None
        self.label_set = None

    def get_labels(self):
        """Gets the list of labels for this data set."""
        if not hasattr(self, 'label_set'):
            raise NotImplementedError()
        else:
            return list(self.label_set.label_list)

    def _create_examples(self, fn, mode):
        raise NotImplementedError()

    def get_examples(self, mode):
        if mode not in ('train', 'test', 'val'):
            raise ValueError(
                (
                    f'Incorrect mode="{mode}", '
                    'expected "train", "test", or "val".'
                )
            )
        if self.data_filenames is None:
            # classes which inherit should define self.data_filenames dictionary
            raise NotImplementedError
        elif mode not in self.data_filenames:
            raise ValueError(
                f'{mode} is not available for {self.__class__.__name__}'
            )

        fn = os.path.join(self.data_dir, self.data_filenames[mode])
        return self._create_examples(fn, mode)

    def _read_file(self, input_file, delimiter=',', quotechar='"'):
        """Reads a comma separated value file."""
        fn = os.path.join(self.data_dir, input_file)
        with open(fn, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return self._read_file(input_file, delimiter='\t', quotechar=quotechar)

    def _read_csv(self, input_file, quotechar='"'):
        """Reads a comma separated value file."""
        return self._read_file(input_file, delimiter=',', quotechar=quotechar)


class DeidProcessor(TokenClassificationTask):
    """Processor for the de-id datasets."""
    def __init__(self, data_dir, label_set, tagger=None):
        """Initialize a data processor with the location of the data."""
        super().__init__(data_dir)
        self.data_filenames = {'train': 'train', 'test': 'test'}
        self.label_set = label_set

        self.tagger = tagger

    def _create_examples(self, fn, set_type):
        """Creates examples for the training, validation, and test sets."""
        examples = []
        fn = os.path.join(fn, set_type.value)

        # for de-id datasets, "fn" is a folder containing txt/ann subfolders
        # "txt" subfolder has text files with the text of the examples
        # "ann" subfolder has annotation files with the labels
        txt_path = os.path.join(fn, 'txt')
        ann_path = os.path.join(fn, 'ann')
        for f in os.listdir(txt_path):
            if not f.endswith('.txt'):
                continue

            # guid = "%s-%s" % (set_type, f[:-4])
            guid = f[:-4]
            with open(os.path.join(txt_path, f), 'r') as fp:
                text = ''.join(fp.readlines())

            # load in the annotations
            fn = os.path.join(ann_path, f'{f[:-4]}.gs')

            # load the labels from a file
            # these datasets have consistent folder structures:
            #   root_path/txt/RECORD_NAME.txt - has text
            #   root_path/ann/RECORD_NAME.gs - has annotations
            self.label_set.from_csv(fn)

            tags = None
            if self.tagger is not None:
                # call a function to generate binary tagging features
                tags = self.tagger(text)

            examples.append(
                InputExample(
                    guid=guid,
                    text=text,
                    labels=self.label_set.labels,
                    patterns=patterns
                )
            )

        return examples

    # methods for token classification task
    def read_examples_from_file(self, data_dir, mode) -> List[InputExample]:
        return self._create_examples(data_dir, mode)

    def write_predictions_to_file(
        self, writer: TextIO, test_input_reader: TextIO, preds_list: List
    ):
        writer.write('\n'.join(preds_list))

    # methods for assigning labels // tokenizing
    def get_token_labels(
        self,
        encoded,
        labels,
        pad_token_label_id=-100,
        default_label='O',
        label_offset_shift=0
    ):
        """
        label_offset_shift: subtract this off label indices. Helps facilitate slicing
        documents into sub-parts.
        """
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
        if labels is None:
            label_ids = [
                self.label_set.label_to_id[default_label]
                for i in range(len(token_labels))
            ]
            return token_labels, label_ids

        offsets = [o[0] for o in encoded.offsets]
        for label in labels:
            entity_type = label.entity_type
            start, offset = label.start, label.length
            if label_offset_shift > 0:
                start -= label_offset_shift
                if start < 0:
                    continue
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

        label_ids = [
            self.label_set.label_to_id[l]
            if l != pad_token_label_id else pad_token_label_id
            for l in token_labels
        ]

        return token_labels, label_ids

    def tokenize_with_labels(
        self, tokenizer, example, pad_token_label_id=-100, default_label='O'
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

        token_labels = self.get_token_labels(
            encoded, example.labels, pad_token_label_id=-100, default_label='O'
        )

        return word_tokens, token_labels, token_sw, offsets, lengths

    def convert_examples_to_features(
        self,
        examples: List[InputExample],
        label_list: List[str],
        tokenizer,
        feature_overlap=None,
        include_offsets=False,
    ):
        """
        Loads a data file into a list of `InputFeatures`s

            `feature_overlap` - Split a single long example into multiple training observations. This is
            useful for handling examples containing very long passages of text.
                None (default): truncates each example at max_seq_length -> one InputFeature per InputExample.
                [0, 1): controls how much overlap between consecutive segments.
                    e.g. `feature_overlap=0.1` means the last 10% of InputFeature 1 will equal first 10%
                    of InputFeature 2, assuming that the InputExample is long enough to require splitting.
        """
        pad_token_label_id = -100
        features = []
        n_obs = 0
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            # tokenize the example text
            encoded = tokenizer._tokenizer.encode(
                example.text, add_special_tokens=False
            )
            token_sw = [False] + [
                encoded.words[i + 1] == encoded.words[i]
                for i in range(len(encoded.words) - 1)
            ]
            token_offsets = np.array(encoded.offsets)

            seq_len = tokenizer.max_len_single_sentence
            if feature_overlap is None:
                feature_overlap = 0
            # identify the starting offsets for each sub-sequence
            new_seq_jump = int((1 - feature_overlap) * seq_len)

            # iterate through subsequences and add to examples
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

                text = example.text[token_offsets[start, 0]:token_offsets[stop,
                                                                          0]]
                encoded = tokenizer._tokenizer.encode(text)
                encoded.pad(tokenizer.model_max_length)

                # assign labels based off the offsets
                _, label_ids = self.get_token_labels(
                    encoded,
                    example.labels,
                    pad_token_label_id=pad_token_label_id,
                    label_offset_shift=token_offsets[start, 0]
                )

                features.append(
                    InputFeatures(
                        input_ids=encoded.ids,
                        attention_mask=encoded.attention_mask,
                        token_type_ids=encoded.type_ids,
                        label_ids=label_ids,
                        # input_offsets=[o[0] for o in encoded.offsets],
                        # input_lengths=[o[1] - o[0] for o in encoded.offsets]
                    )
                )
                n_obs += 1

                # update start of next sequence to be end of current one
                start = start + new_seq_jump

        return features
