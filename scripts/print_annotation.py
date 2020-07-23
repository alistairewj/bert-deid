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

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
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

    # file to run
    parser.add_argument(
        '-t',
        "--text",
        default=None,
        required=True,
        type=str,
        help=("Text file to assess.")
    )

    # gold standard params
    parser.add_argument(
        '-r',
        "--ref",
        default=None,
        type=str,
        help=("The input reference file.")
    )

    parser.add_argument(
        '-p',
        "--pred",
        default=None,
        type=str,
        help="Optionally compare to a model."
    )

    parser.add_argument(
        '-b',
        '--brat_path',
        type=str,
        default=None,
        help='folder to output brat annotations'
    )

    args = parser.parse_args()

    for fn in [args.text, args.pred, args.ref]:
        if not os.path.exists(fn):
            raise ValueError(f'{fn} is not a valid file.')

    fn = args.text
    fn_pred = args.pred
    fn_gs = args.ref

    # load the text
    with open(fn, 'r') as fp:
        text = ''.join(fp.readlines())

    # load output of bert-deid
    df = pd.read_csv(
        fn_pred, header=0, dtype={
            'entity': str,
            'entity_type': str
        }
    )

    # load ground truth
    gs = pd.read_csv(fn_gs, header=0, dtype={'entity': str, 'entity_type': str})

    # binary vector indicating PHI/not phi
    text_tar = np.zeros(len(text), dtype=bool)
    for i, row in gs.iterrows():
        text_tar[row['start']:row['stop']] = True

    # binary vector indicating if predicted as phi
    if args.pred is None:
        # print only green for PHI
        text_pred = text_tar
    else:
        text_pred = np.zeros(len(text), dtype=bool)
        for i, row in df.iterrows():
            text_pred[row['start']:row['stop']] = True

    i = 0
    # inform the user of the legend
    print('\n')
    print(
        f'{bcolors.OKGREEN}TRUE POSITIVE.{bcolors.ENDC} If no predictions given, PHI will be annotated in this color.'
    )
    print(f'{bcolors.FAIL}FALSE NEGATIVE.{bcolors.ENDC}')
    print(f'{bcolors.OKBLUE}FALSE POSITIVE.{bcolors.ENDC}\n')
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
