# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2019, Alistair Johnson.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import itertools

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

from seqeval.metrics import classification_report
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForTokenClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# custom tokenizer with subword tracking
from tokenization import BertTokenizerNER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def segment_ids(self, segment1_len, segment2_len):
    ids = [0] * segment1_len + [1] * segment2_len
    return torch.tensor([ids]).to(device=self.device)


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


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file, delimiter=',', quotechar='"'):
        """Reads a comma separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return cls._read_file(input_file, delimiter='\t', quotechar=quotechar)

    @classmethod
    def _read_csv(cls, input_file, quotechar='"'):
        """Reads a comma separated value file."""
        return cls._read_file(input_file, delimiter=',', quotechar=quotechar)


class DeidProcessor(DataProcessor):
    """Processor for the de-id datasets with harmonized labels."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return [
            # names
            'NAME',  # 'DOCTOR', 'PATIENT', 'USERNAME',
            # professions
            'PROFESSION',
            # locations
            'LOCATION',  # 'HOSPITAL', 'ORGANIZATION',
            # 'STREET', 'STATE', 'CITY', 'COUNTRY', 'ZIP',
            # 'LOCATION-OTHER',
            # ages, including those over 89
            'AGE',
            # dates
            'DATE',
            # IDs
            'ID',
            # 'BIOID', 'DEVICE',  'HEALTHPLAN',  'IDNUM', 'MEDICALRECORD',
            'CONTACT',
            # 'EMAIL', 'FAX', 'PHONE', 'URL', 'IPADDRESS'
            # catch all object
            'O'
        ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            # dataset has stored labels as a list of offsets
            # trust the data and call eval on the string
            label = eval(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class i2b2Processor(DataProcessor):
    """Processor for the i2b2 de-id data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return [
            # names
            'DOCTOR', 'PATIENT', 'USERNAME',
            # professions
            'PROFESSION',
            # locations
            'HOSPITAL', 'ORGANIZATION',
            'STREET', 'STATE', 'CITY', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
            # age > 89
            'AGE',
            # dates
            'DATE',
            # IDs
            'BIOID', 'DEVICE',  'HEALTHPLAN',  'IDNUM', 'MEDICALRECORD',
            'EMAIL', 'FAX', 'PHONE', 'URL',
            # catch all object
            'O'
        ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            # dataset has stored labels as a list of offsets
            # trust the data and call eval on the string
            label = eval(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class CoNLLProcessor(DataProcessor):
    """Processor for the gold standard de-id data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['[CLS]', '[SEP]', 'O',
                'B-LOC', 'B-MISC', 'B-ORG', 'B-PER',
                'I-LOC', 'I-MISC', 'I-ORG', 'I-PER']
        # return ['[CLS]', '[SEP]', 'LOC', 'MISC', 'ORG', 'PER', 'O']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence = []
        label = []
        # n keeps track of the number of samples processed
        n = 0
        # guid = [dataset - sample number - starting line number in original file]
        guid = "%s-%s-%s" % (set_type, n, 2)
        for (i, line) in enumerate(lines):
            if len(line) == 0:
                if len(sentence) == 0:
                    continue
                # end of sentence

                # reformat labels to be a list of anns: [start, stop, entity]
                # e.g. [[0, 2, 'O'], [2, 6, 'ORG], ...]
                label_offsets = []
                s_len = 0
                for j, l in enumerate(sentence):
                    # label_new = combine_labels[label[j]]
                    label_new = label[j]
                    label_offsets.append(
                        [s_len, s_len+len(l), label_new])
                    # +1 to account for the whitespaces we insert below
                    s_len += len(l) + 1

                # create a single string for the sentence
                sentence = ' '.join(sentence)
                examples.append(
                    InputExample(guid=guid, text_a=sentence, label=label_offsets))
                sentence = []
                label = []
                n += 1
                guid = "%s-%s-%s" % (set_type, n, i+1)
                continue

            line = line[0]
            if line.startswith('-DOCSTART-'):
                continue

            text = line.split(' ')
            sentence.append(text[0])
            label.append(text[3])
        return examples


def prepare_tokens(tokens, tokens_sw, tokens_idx, label,
                   label_list, max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}

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
        tokens_a, tokens_a_sw, tokens_a_idx = tokenizer.tokenize(
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
        segment_ids = [0] * len(tokens)

        input_ids, input_mask, segment_ids, label_ids = prepare_tokens(
            tokens, tokens_sw, tokens_idx, example.label,
            label_list, max_seq_length, tokenizer)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" %
                        " ".join([str(x) for x in label_ids]))

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


def accuracy(out, labels):
    outputs = np.argmax(out, axis=-1)
    return np.sum(outputs == labels)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--finetune",
                        action='store_true',
                        help="Fine-tune top layer only.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")

    # allow output of predictions
    parser.add_argument('--output_predictions', action='store_true',
                        help="Output predictions to file")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "deid": DeidProcessor,
        "conll": CoNLLProcessor,
        "i2b2": i2b2Processor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizerNER.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForNER.from_pretrained(args.bert_model,
                                       cache_dir=cache_dir,
                                       num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # if fine-tuning, do not pass bert parameters to optimizer
    if args.finetune:
        param_optimizer = [(n, p) for n, p in param_optimizer
                           if not n.startswith('bert.')]
        logger.info(("Only training classification layers "
                     "({} modules, {} parameters).").format(
            len(param_optimizer), sum([len(p) for n, p in param_optimizer])))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()

        if args.finetune:
            # freeze lower levels of bert encoder
            for p in model.bert.parameters():
                p.requires_grad = False

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * \
                            warmup_linear(
                                global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
    else:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        if os.path.exists(output_model_file) & os.path.exists(output_config_file):
            # Load a trained model and config that you have fine-tuned
            print(f'Loading model and configuration from {args.output_dir}.')
            config = BertConfig(output_config_file)
            model = BertForNER(config, num_labels=num_labels)
            model.load_state_dict(torch.load(output_model_file))
        else:
            print('No trained model/config found in output_dir.')
            print('Using pretrained BERT model with random classification weights.')
            model = BertForNER.from_pretrained(
                args.bert_model, num_labels=num_labels)

        model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # create result dictionaries for all labels
        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        label_id_map = {i: label for i, label in enumerate(label_list)}
        tp = {x: 0 for x in label_map}
        fp = {x: 0 for x in label_map}
        tn = {x: 0 for x in label_map}
        fn = {x: 0 for x in label_map}
        n_samples = {x: 0 for x in label_map}

        y_pred = []
        y_true = []

        # keep track of IDs so we can output predictions to file with GUID
        # requires us to use sequential sampler
        if args.output_predictions:
            assert type(eval_sampler) == SequentialSampler, \
                'Must use sequential sampler if outputting predictions'

            output_fn = os.path.join(args.output_dir, "eval_predictions.csv")
            logger.info("***** Outputting predictions to %s *****", output_fn)
            fp_output = open(output_fn, 'w')
            out_writer = csv.writer(fp_output, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            n = 0

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(
                    input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            idx = (input_mask.to('cpu').numpy() == 1) & (label_ids != -1)
            yhat = np.argmax(logits, axis=-1)

            # add this batch to the y_true/y_pred lists
            for j in range(yhat.shape[0]):
                yhat_row = yhat[j, :]
                y_row = label_ids[j, :]
                y_true.append([label_id_map[y]
                               for i, y in enumerate(y_row)
                               if idx[j, i] & (y_row[i] != -1)])
                y_pred.append([label_id_map[y]
                               for i, y in enumerate(yhat_row)
                               if idx[j, i] & (y_row[i] != -1)])

                # remove [CLS] and [SEP] tags
                y_pred[-1] = [x for i, x in enumerate(y_pred[-1])
                              if y_true[-1][i] not in ("[CLS]", "[SEP]")]
                y_true[-1] = [x for i, x in enumerate(y_true[-1])
                              if y_true[-1][i] not in ("[CLS]", "[SEP]")]

            if args.output_predictions:
                for j in range(yhat.shape[0]):
                    out_writer.writerow(
                        [eval_examples[n+j].guid] + list(yhat[j, :]))
                n += 8

            # calculate running total for true positives, false positives, etc.
            for lbl in label_map:
                lbl_id = label_map[lbl]
                tp[lbl] += (idx &
                            (yhat == lbl_id) & (label_ids == lbl_id)).sum()
                fp[lbl] += (idx &
                            (yhat == lbl_id) & (label_ids != lbl_id)).sum()
                tn[lbl] += (idx &
                            (yhat != lbl_id) & (label_ids != lbl_id)).sum()
                fn[lbl] += (idx &
                            (yhat != lbl_id) & (label_ids == lbl_id)).sum()
                n_samples[lbl] += ((label_ids == lbl_id) & idx).sum()

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        if args.output_predictions:
            fp_output.close()

        eval_loss = eval_loss / nb_eval_steps
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result = {'eval_loss': eval_loss,
                  'global_step': global_step,
                  'loss': loss}

        # append label-wise metrics
        stats = {}
        stats['f1'], stats['se'], stats['p+'] = {}, {}, {}

        for lbl in label_map:
            lbl_id = label_map[lbl]
            stats['se'][lbl] = tp[lbl]/(tp[lbl] + fn[lbl])
            stats['p+'][lbl] = tp[lbl]/(tp[lbl] + fp[lbl])
            stats['f1'][lbl] = (2*tp[lbl])/(2*tp[lbl] + fn[lbl] + fp[lbl])
            result[lbl + '_f1'] = stats['f1'][lbl]

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # print out stats from running tallies
        logger.info('{:10s} {:10s} {:10s} {:10s} {:10s}'.format(
            '', 'ppv', 'sens', 'f1', 'num'
        ))
        for lbl in label_map:
            lbl_id = label_map[lbl]
            logger.info('{:10s} {:10.4f} {:10.4f} {:10.4f} {:10d}'.format(
                lbl,
                stats['p+'][lbl], stats['se'][lbl], stats['f1'][lbl],
                n_samples[lbl]
            ))

        # print out stats from seqeval package
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("classification report")
        logger.info("\n%s", report)


if __name__ == "__main__":
    main()
