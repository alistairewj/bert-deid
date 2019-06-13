from __future__ import absolute_import, division, print_function

import json
import os
import re
import logging
import argparse

from bert_deid import utils
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('-c', "--config",
                        default=None,
                        type=str,
                        help=("Configuration file (json)."
                              "Specifies folders and parameters."))

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

    csv_path = None
    if argparse_dict['csv_path'] is not None:
        csv_path = argparse_dict['csv_path']
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

    # ensure extension vars have the dot prefix
    for c in argparse_dict.keys():
        if c.endswith('_extension'):
            if argparse_dict[c][0] != '.':
                argparse_dict[c] = '.' + argparse_dict[c]

    input_ext = argparse_dict['text_extension']
    gs_ext = argparse_dict['ref_extension']
    pred_ext = argparse_dict['pred_extension']

    # read files from folder
    if os.path.exists(argparse_dict['text_path']):
        input_path = argparse_dict['text_path']
        files = os.listdir(input_path)

        # remove extension and get file list
        input_files = set([f[0:-len(input_ext)]
                           for f in files
                           if f.endswith(input_ext)])

        input_files = list(input_files)
    else:
        raise ValueError('Input folder %s does not exist.',
                         argparse_dict['text_path'])

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(input_files))

    perf_all = {}
    for fn in tqdm(input_files, total=len(input_files)):
        # load the text
        with open(os.path.join(input_path, f'{fn}{input_ext}'), 'r') as fp:
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

        curr_performance = {}
        curr_performance['n_char'] = len(text)
        curr_performance['n_char_phi'] = sum(text_tar)
        curr_performance['n_char_tp'] = sum((text_tar & text_pred))
        curr_performance['n_char_fp'] = sum((~text_tar & text_pred))
        curr_performance['n_char_fn'] = sum((text_tar & ~text_pred))

        # report performance token wise
        tokens, tokens_y, tokens_ypred = [], [], []
        pattern = re.compile(r'\s')
        n_tokens = 0
        for token, start, end in utils.pattern_spans(text, pattern):
            token_tar = False
            token_pred = False

            # if any of the individual characters are flagged
            # .. then we consider the whole token to be flagged
            for i in range(start, end):
                if text_tar[i]:
                    token_tar = True
                if text_pred[i]:
                    token_pred = True

            if token_tar | token_pred:
                tokens.append([token, start, end])
                tokens_y.append(token_tar)
                tokens_ypred.append(token_pred)

            n_tokens += 1

        # now we have a list of tokens with preds, calculate some stats
        tokens_y = np.asarray(tokens_y, dtype=bool)
        tokens_ypred = np.asarray(tokens_ypred, dtype=bool)

        curr_performance['n_token'] = n_tokens
        curr_performance['n_token_phi'] = sum(tokens_y)
        curr_performance['n_token_tp'] = sum(tokens_y & tokens_ypred)
        curr_performance['n_token_fp'] = sum(~tokens_y & tokens_ypred)
        curr_performance['n_token_fn'] = sum(tokens_y & ~tokens_ypred)

        perf_all[fn] = curr_performance

    # convert to dataframe
    df = pd.DataFrame.from_dict(perf_all, orient='index')

    print(df)

    if csv_path is not None:
        df.to_csv(os.path.join(csv_path, 'performance.csv'))


if __name__ == "__main__":
    main()
