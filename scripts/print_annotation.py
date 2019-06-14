from __future__ import absolute_import, division, print_function

import json
import os
import logging
import argparse
import sys

from bert_deid import utils

import pandas as pd
import numpy as np
"""
Runs BERT deid on a set of text files.
Evaluates the output using gold standard annotations.

Optionally outputs mismatches to brat standoff format.
"""


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('-c', "--config",
                        default=None,
                        type=str,
                        help=("Configuration file (json)."
                              "Specifies folders and parameters."))
    # file to run
    parser.add_argument('-f', "--filename",
                        default=None,
                        required=True,
                        type=str,
                        help=("Filename to run bert deid on."))

    # If config is not provided, each argument can be given as an input
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        help="The directory with model weights/config files.")

    parser.add_argument("--text_path",
                        default=None,
                        type=str,
                        help=("The input data folder with individual"
                              " files to be deidentified."))
    parser.add_argument("--text_extension",
                        default='txt',
                        type=str,
                        help=("Extension for input files in input folder "
                              "(default: txt)"))

    # gold standard params
    parser.add_argument("--ref_path",
                        default=None,
                        type=str,
                        help=("The input ref folder with labels"
                              " for text files to be deidentified."))
    parser.add_argument("--ref_extension",
                        default='gs',
                        type=str,
                        help=("Extension for gold standard files in"
                              " ref folder (default: gs)"))

    parser.add_argument("--pred_orig_path",
                        default=None,
                        type=str,
                        help="Path to output unsimplified model labels.")
    parser.add_argument("--pred_path",
                        default=None,
                        type=str,
                        help="Path to output simplified model labels.")
    parser.add_argument("--pred_extension",
                        default='bert',
                        type=str,
                        help=("Extension for output labels"
                              " (default: bert)"))

    parser.add_argument('-b', '--brat_path', type=str,
                        default=None,
                        help='folder to output brat annotations')

    parser.add_argument('--csv_path', type=str,
                        default=None,
                        help='folder to output errors for labeling')

    # parameters of the model/deid
    parser.add_argument("--task_name",
                        default='i2b2',
                        type=str,
                        choices=['i2b2', 'hipaa'],
                        help=("The name of the task to train. "
                              "Primarily defines the label set."))

    parser.add_argument('-m', '--method', type=str,
                        default='sentence',
                        choices=['sentence', 'overlap'],
                        help=('method for splitting text into'
                              ' individual examples.'))
    parser.add_argument('--step-size', type=int,
                        default=20,
                        help='if method="overlap", the token step size to use.')
    parser.add_argument('--sequence-length', type=int,
                        default=100,
                        help='if method="overlap", the max number of tokens per ex.')
    # Other parameters
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    # if config file is used, we ignore args and use config file/defaults
    # TODO: allow using environment variables?

    # prepare a dict with values from argparse (incl defaults)
    argparse_dict = vars(args)
    if args.config is not None:
        # if config provided, update dict with values from config
        with open(args.config, 'r') as fp:
            config = json.load(fp)

        argparse_dict.update(config)

    # if pred_orig_path exists, we are outputting non-simplified annot
    # if neither exist, warn the user that we are not saving ann to files
    if argparse_dict['pred_path'] is None:
        raise ValueError('Prediction path required to evaluate predictions.')

    # ensure extension vars have the dot prefix
    for c in argparse_dict.keys():
        if c.endswith('_extension'):
            if argparse_dict[c][0] != '.':
                argparse_dict[c] = '.' + argparse_dict[c]

    input_ext = argparse_dict['text_extension']
    gs_ext = argparse_dict['ref_extension']
    pred_ext = argparse_dict['pred_extension']

    fn = args.filename
    # load the text
    with open(os.path.join(argparse_dict['text_path'], f'{fn}{input_ext}'), 'r') as fp:
        text = ''.join(fp.readlines())

    # load output of bert-deid
    fn_pred = f'{fn}{pred_ext}'
    df = pd.read_csv(os.path.join(argparse_dict['pred_path'], fn_pred),
                     header=0)

    # load ground truth
    gs_fn = os.path.join(argparse_dict['ref_path'], f'{fn}{gs_ext}')
    gs = pd.read_csv(gs_fn, header=0,
                     dtype={'entity': str,
                            'entity_type': str})

    # fix entities - lower case and group
    gs = utils.combine_entity_types(gs, lowercase=True)

    # binary vector indicating PHI/not phi
    text_tar = np.zeros(len(text), dtype=bool)
    for i, row in gs.iterrows():
        text_tar[row['start']:row['stop']] = True

    # binary vector indicating if predicted as phi
    text_pred = np.zeros(len(text), dtype=bool)
    for i, row in df.iterrows():
        text_pred[row['start']:row['stop']] = True

    i = 0
    print('\n\n')
    color = 'black'
    while i < len(text):

        if text_tar[i] & text_pred[i]:
            if color != 'green':
                print(bcolors.OKGREEN, end='')
                print(bcolors.BOLD, end='')
                color = 'green'
        elif text_tar[i]:
            if color != 'red':
                print(bcolors.FAIL, end='')
                print(bcolors.BOLD, end='')
                color = 'red'
        elif text_pred[i]:
            if color != 'yellow':
                print(bcolors.OKBLUE, end='')
                print(bcolors.BOLD, end='')
                color = 'yellow'
        else:
            if color != 'black':
                print(bcolors.ENDC, end='')
                color = 'black'

        # make it obvious if we have annotated spaces by printing underscore
        if color == 'black':
            print(text[i], end='')
        elif text[i] in (' ', '\r', '\n'):
            print('_', end='')
            if text[i] == '\n':
                print('')
        else:
            print(text[i], end='')
        i += 1

        sys.stdout.flush()

    print(bcolors.ENDC)
    print('\n\n')


if __name__ == "__main__":
    main()
