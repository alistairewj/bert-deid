"""Applies a BERT NER model to a text file or folder of text files."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import sys
import itertools

# must have installed punkt model
# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# custom tokenizer with subword tracking
from tokenization import BertTokenizerNER

import bert_ner
from bert_ner import prepare_tokens, BertForNER, InputFeatures
from create_csv import split_by_overlap, split_by_sentence
from utils import harmonize_label

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input",
                        default=None,
                        type=str,
                        required=True,
                        help=("The input data. Either a single text file, "
                              "or a folder containing .txt files"))
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the model checkpoints are.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        choices=[
                            "bert-base-uncased", "bert-base-cased",
                            "bert-large-uncased", "bert-large-cased",
                            "bert-base-multilingual-uncased",
                            "bert-base-multilingual-cased",
                            "bert-base-chinese"],
                        help="BERT pre-trained model")
    parser.add_argument("--task_name",
                        default='i2b2',
                        required=True,
                        type=str,
                        choices=['i2b2', 'hipaa'],
                        help=("The name of the task to train. "
                              "Primarily defines the label set."))

    parser.add_argument('-m', '--method', type=str,
                        default='sentence',
                        choices=['sentence', 'overlap'],
                        help='method for splitting text into individual examples.')
    parser.add_argument('--step-size', type=int,
                        default=20,
                        help='if method is overlap, the token step size to use.')
    parser.add_argument('--sequence-length', type=int,
                        default=100,
                        help='if method is overlap, the maximum token length.')
    # Other parameters
    parser.add_argument("--labels",
                        default=None,
                        type=str,
                        help=("The input labels. Either a single gs file, "
                              "or a folder containing .gs files"))
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--conll', action='store_true',
                        help="Output predictions in CoNLL format.")

    args = parser.parse_args()

    processors = {
        "deid": bert_ner.DeidProcessor,
        "conll": bert_ner.CoNLLProcessor,
        "i2b2": bert_ner.i2b2Processor
    }

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizerNER.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    # load trained model
    output_model_file = os.path.join(args.model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.model_dir, CONFIG_NAME)
    if os.path.exists(output_model_file) & os.path.exists(output_config_file):
        # Load a trained model and config that you have fine-tuned
        print(f'Loading model and configuration from {args.model_dir}.')
        config = BertConfig(output_config_file)
        model = BertForNER(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        raise ValueError('Folder %s did not have model and config file.',
                         args.model_dir)

    model.to(device)

    # if input is a single text file, wrap it in a list
    if os.path.exists(args.input):
        if os.path.isdir(args.input):
            input_path = args.input
            label_path = args.labels
            files = os.listdir(input_path)
            files = [f for f in files if f.endswith('.txt')]

            label_files = os.listdir(label_path)
            label_files = [f for f in label_files if f.endswith('.gs')]

            # ensure we have files for each input/label
            doc_id = [f[0:-4] for f in files if f[0:-4] + '.gs' in label_files]
            label_files = [f + '.gs' for f in doc_id]
            files = [f + '.txt' for f in doc_id]
        else:
            files = [args.input]
            label_files = [args.labels]
            doc_id = [args.input]
            input_path = ''
            label_path = ''
    else:
        raise ValueError('Input file/folder %s does not exist.',
                         args.input)

    logger.info("Parsing {} input file(s)".format(len(files)))
    tokens_all, tokens_sw_all, tokens_idx_all = [], [], []
    doc_id_all = []

    # create features
    eval_features = []

    # load text
    for i, f in enumerate(doc_id):
        with open(os.path.join(input_path, f + '.txt'), 'r') as fp:
            text = ''.join(fp.readlines())

        # load labels into list
        labels = []
        with open(os.path.join(label_path, f + '.gs'), 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            # skip header
            next(csvreader)
            for row in csvreader:
                # get start/stop/entity type
                labels.append([int(row[2]), int(row[3]), row[5]])

        # if requested, harmonize tags using fixed dictionary
        if args.group_tags:
            # for each example
            for i in range(len(labels)):
                labels[i][2] = harmonize_label(
                    labels[i][2], grouping='hipaa')

        # split the text into sentences
        # this is a list of lists, each sub-list has 4 elements:
        #   sentence number, start index, end index, text of the sentence
        if args.method == 'sentence':
            examples = split_by_sentence(text)
        else:
            examples = split_by_overlap(
                text, tokenizer,
                token_step_size=args.step_size,
                max_seq_len=args.sequence_length
            )

        for e, example in enumerate(examples):
            # track offsets in tokenization
            tokens, tokens_sw, tokens_idx = tokenizer.tokenize(
                example[3])

            # offset labels based upon example start
            start = example[1]
            labels_offset = []
            for k in range(len(labels)):
                if labels[k][1] < start:
                    continue
                # if this label annotates tokens before this example
                if labels[k][0] < start:
                    labels_offset.append([
                        0,
                        labels[k][1]-start,
                        labels[k][2]
                    ])
                else:
                    labels_offset.append([
                        labels[k][0] - start,
                        labels[k][1] - start,
                        labels[k][2]
                    ])

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens_sw = [0] + tokens_sw + [0]
            tokens_idx = [[-1]] + tokens_idx + [[-1]]

            input_ids, input_mask, segment_ids, label_ids = prepare_tokens(
                tokens, tokens_sw, tokens_idx,
                labels_offset, label_list,
                args.max_seq_length, tokenizer
            )

            tokens_all.append(tokens)
            tokens_sw_all.append(tokens_sw)
            tokens_idx_all.append(tokens_idx)
            doc_id_all.append(f)

            eval_features.append(InputFeatures(input_ids=input_ids,
                                               input_mask=input_mask,
                                               segment_ids=segment_ids,
                                               label_ids=label_ids))

            if (i == 0) & (e < 5):
                logger.info("*** Example ***")
                logger.info("guid: %s" % (f'{f}-{e}'))
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

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_features))
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

    # create result dictionaries for all labels
    label_list = processor.get_labels()
    label_id_map = {i: label for i, label in enumerate(label_list)}

    y_pred = []

    # keep track of IDs so we can output predictions to file with GUID
    # requires us to use sequential sampler
    assert type(eval_sampler) == SequentialSampler, \
        'Must use sequential sampler if outputting predictions'
    n = 0

    if args.conll:
        output_fn = "preds.conll"
        logger.info(
            "***** Outputting CoNLL format predictions to %s *****", output_fn)
        fp_output = open(output_fn, 'w')

    for input_ids, input_mask, segment_ids, label_ids in tqdm(
            eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        idx = (input_mask.to('cpu').numpy() == 1) & (label_ids != -1)
        yhat = np.argmax(logits, axis=-1)

        # convert predictions into doc_id, start, stop, type

        # add this batch to the y_true/y_pred lists
        for j in range(yhat.shape[0]):
            tokens = tokens_all[n+j]
            tokens_sw = tokens_sw_all[n+j]
            tokens_idx = tokens_idx_all[n+j]
            tokens_mask = list(idx[j, :])
            labels = label_ids[j, :].tolist()
            pred = yhat[j, :].tolist()

            # remove [CLS] at beginning
            tokens = tokens[1:]
            tokens_sw = tokens_sw[1:]
            tokens_mask = tokens_mask[1:]
            labels = labels[1:]
            pred = pred[1:]

            # subselect pred and mask down to length of tokens
            last_token = tokens.index('[SEP]')
            tokens = tokens[:last_token]
            tokens_sw = tokens_sw[:last_token]
            tokens_mask = tokens_mask[:last_token]
            labels = labels[:last_token]
            pred = pred[:last_token]

            if len(tokens) == 0:
                # no data for predictions
                continue

            pred = [label_id_map[x] for x in pred]

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
                    pred.pop(k)
                    tokens_sw.pop(k)
                    labels.pop(k)
                    tokens_mask.pop(k)
                else:
                    k += 1

            for k in range(len(pred)):
                # skip if masked/subword
                if not tokens_mask[k]:
                    continue

                # for this sentence, get start/stop indices
                start = tokens_idx[k][0]
                stop = tokens_idx[k][-1] + 1

                if args.conll:
                    # output conll format
                    row = [
                        tokens[k],
                        doc_id_all[n+j],
                        str(start),
                        str(stop),
                        label_id_map[labels[k]],
                        pred[k]
                    ]
                    fp_output.write(' '.join(row) + '\n')

                if pred[k] == 'O':
                    continue

                y_pred.append(
                    [doc_id_all[n+j], start, stop, tokens[k], pred[k]])

            if args.conll:
                # add extra blank line
                fp_output.write('\n')

        n += input_ids.shape[0]

    # close conll file
    if args.conll:
        fp_output.close()

    output_fn = os.path.join("bert_preds")
    logger.info("***** Outputting predictions to %s *****", output_fn)
    with open(output_fn, 'w') as fp_output:
        out_writer = csv.writer(fp_output, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in y_pred:
            out_writer.writerow(row)


if __name__ == "__main__":
    main()
