# create a single CSV for training using data stored as gs/txt files
import argparse
import os
import sys
import csv
import itertools

# must have installed punkt model
# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from tqdm import tqdm

from bert_deid.tokenization import BertTokenizerNER
from bert_deid.describe_data import harmonize_label


def argparser(args):
    parser = argparse.ArgumentParser(description='Convert i2b2 annotations')
    parser.add_argument('-i', '--input', type=str,
                        default=None, required=True,
                        help='folder with ann and txt subfolders')
    parser.add_argument('-o', '--output', type=str,
                        default=None, required=True,
                        help=('Filename of CSV to output all data to.'
                              ' (default: do not create a CSV.)'))

    # optional arguments
    parser.add_argument("--task_name",
                        default='i2b2',
                        type=str,
                        choices=['i2b2', 'hipaa'],
                        help="Defines the label set.")
    parser.add_argument('-g', '--group-tags', action='store_true',
                        help='group tags into categories.')
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
    parser.add_argument("--bert_model", type=str,
                        default="bert-base-cased",
                        choices=[
                            "bert-base-uncased", "bert-base-cased",
                            "bert-large-uncased", "bert-large-cased",
                            "bert-base-multilingual-uncased",
                            "bert-base-multilingual-cased",
                            "bert-base-chinese"],
                        help="BERT pre-trained model for tokenization")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress peasants discussing their work')

    return parser.parse_args(args)


# our dataframe will have consistent columns
COLUMN_NAMES = ['document_id', 'annotation_id', 'start',
                'stop', 'entity', 'entity_type', 'comment']


def sentence_spans(text):
    """
    Iterator that splits text into sentences.
    Also returns the span of sentences (start and stop indices).
    """
    tokens = sent_tokenize(text)

    # further split using token '\n \n'
    tokens = [x.split('\n \n') for x in tokens]
    # flatten sublists into a single list
    tokens = list(itertools.chain.from_iterable(tokens))

    # further split using token '\n\n'
    tokens = [x.split('\n\n') for x in tokens]
    tokens = list(itertools.chain.from_iterable(tokens))

    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)


def split_by_sentence(text):
    """
    Iterator that wraps nltk punkt sentence splitter.
    """
    # section the reports
    # p_section = re.compile('^\s+([A-Z /,-]+):', re.MULTILINE)
    sentences = list()
    n = 0
    for sent, start, end in sentence_spans(text):
        sentences.append([n, start, end, sent])
        n += 1

    return sentences


def split_by_overlap(text, tokenizer,
                     token_step_size=20, max_seq_len=100):
    # track offsets in tokenization
    tokens, tokens_sw, tokens_idx = tokenizer.tokenize(text)

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

    if len(tokens) <= max_seq_len:
        # very short text - only create one example
        seq_offsets = [[tokens_start[0], len(text)]]
    else:
        seq_offsets = range(0, len(tokens) - max_seq_len, token_step_size)
        last_offset = seq_offsets[-1] + token_step_size
        seq_offsets = [[tokens_start[x], tokens_start[x + max_seq_len]]
                       for x in seq_offsets]

        # last example always goes to the end of the text
        seq_offsets.append([tokens_start[last_offset], len(text)])

    # turn our offsets into examples
    # create a list of lists, each sub-list has 4 elements:
    #   sentence number, start index, end index, text of the sentence
    examples = list()

    for i, (start, stop) in enumerate(seq_offsets):
        examples.append([i, start, stop, text[start:stop]])

    return examples


def create_ann(examples, df):
    annotations = []
    d_start = 0
    for i in range(len(examples)):
        start, stop = examples[i][1:3]

        # add de-id info
        deid_info = []
        # we rely on de-id being sorted in order of appearance
        d = d_start

        for _, row in df.iloc[d:, :].iterrows():
            # check if we need to update d_start
            if row['stop'] < start:
                # if the de-id annot ends before the text starts
                d_start = d
                d += 1
                continue

            # check if we have finished assigning deid
            if row['start'] > stop:
                break

            if row['start'] >= start:
                if row['stop'] <= stop:
                    # add indices of annotated de-id
                    # adjust them based upon example start offset
                    deid_info.append([
                        row['start'] - start,
                        row['stop'] - start,
                        row['entity_type']
                    ])
                else:
                    # if we are here, the PHI row starts in this example..
                    # but ends in the next example. this is an edge case.
                    # annotate up to the end of this example
                    deid_info.append([
                        row['start'] - start,
                        stop - start,
                        row['entity_type']
                    ])

            d += 1

        annotations.append(deid_info)

    return annotations


def main(args):
    args = argparser(args)

    input_path = args.input
    output_fn = args.output
    verbose_flag = not args.quiet

    # check input folders exist
    ann_path = os.path.join(input_path, 'ann')
    txt_path = os.path.join(input_path, 'txt')
    if not os.path.exists(input_path):
        raise ValueError(f'Could not find folder {input_path}')
    if not os.path.exists(ann_path):
        raise ValueError(f'Could not find folder {ann_path}')
    if not os.path.exists(txt_path):
        raise ValueError(f'Could not find folder {txt_path}')

    ann_files = os.listdir(ann_path)
    txt_files = os.listdir(txt_path)

    # filter to files with correct extension // that overlap
    ann_files = [f for f in ann_files if f.endswith('.gs')]
    txt_files = set([f for f in txt_files if f.endswith('.txt')])
    records = [f[0:-3] for f in ann_files if f[0:-3] + '.txt' in txt_files]

    if len(records) == 0:
        print(f'No files with both text and annotations.')
        return

    examples = list()
    annotations = list()

    if verbose_flag:
        N = len(records)
        print(f'Processing {N} records found.')
        records = tqdm(records)

    n = 0
    n_ex = 0
    n_ex_filtered = 0

    if args.method == 'overlap':
        # initialize bert tokenizer
        if 'uncased' in args.bert_model:
            do_lower_case = True
        else:
            do_lower_case = False

        tokenizer = BertTokenizerNER.from_pretrained(
            args.bert_model, do_lower_case=do_lower_case)

    for doc_id in records:
        # read PHI annotations
        df = pd.read_csv(os.path.join(ann_path, doc_id + '.gs'))
        with open(os.path.join(txt_path, doc_id + '.txt'), 'r') as fp:
            text = ''.join(fp.readlines())

        # keep track of number of records processed
        n += 1

        # filter text which is entirely newline/space
        if text.isspace():
            continue

        # split the text into individual examples
        # this creates a list of lists, each sub-list has 4 elements:
        #   example number, start index, end index, text of the example
        if args.method == 'sentence':
            example = split_by_sentence(text)
        else:
            example = split_by_overlap(
                text, tokenizer,
                token_step_size=args.step_size,
                max_seq_len=args.sequence_length
            )

        example_ann = create_ann(example, df)

        len_example = len(example)

        # filter examples which are entirely newline/space
        idxKeep = [(len(x[3]) > 0) & (~x[3].isspace()) for x in example]
        example = [x for i, x in enumerate(example) if idxKeep[i]]
        example_ann = [x for i, x in enumerate(example_ann) if idxKeep[i]]

        # keep track of removed tokens
        n_ex += len_example
        n_ex_filtered = len_example - len(example)

        if len(example) == 0:
            continue

        # add document_id to examples generated
        example = [[doc_id] + x for x in example]

        # if requested, harmonize tags using fixed dictionary
        if args.group_tags:
            # for each example
            for i in range(len(example_ann)):
                # for each entity tagged in this example
                # (if no entities, this for loop does not run)
                for k in range(len(example_ann[i])):
                    # update the entity (element 2) using label_to_type dict
                    example_ann[i][k][2] = harmonize_label(
                        example_ann[i][k][2], grouping=args.task_name)
        # save to master list
        examples.extend(example)
        annotations.extend(example_ann)

    if verbose_flag:
        print(f'Successfully read {n} records.')
        print(f'  {n_ex} examples were initially created.')
        print(f'  {n_ex_filtered} examples were filtered as they were empty.')
        print('Final dataset size: {}'.format(len(examples)))

    # create sentence-wise deid format with all sentence data
    output_dir = os.path.dirname(output_fn)
    if not os.path.exists(output_dir):
        if verbose_flag:
            print(f'Creating directory {output_dir}')

        os.makedirs(output_dir)

    if verbose_flag:
        print(f'Outputting CSV dataset to {output_fn}')

    with open(output_fn, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(annotations)):
            sentence = examples[i]
            # create row: text id, sentence, list of PHI indices
            row = ['.'.join([str(x) for x in sentence[:4]]),
                   sentence[4], annotations[i]]
            csvwriter.writerow(row)


if __name__ == '__main__':
    main(sys.argv[1:])
