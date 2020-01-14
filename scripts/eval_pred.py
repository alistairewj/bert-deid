from __future__ import absolute_import, division, print_function

import json
from collections import OrderedDict
import os
import re
import logging
import argparse
from pathlib import Path

from bert_deid import utils
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path", required=True, type=str, help="Path for model labels."
    )
    parser.add_argument(
        "--text_path",
        required=True,
        type=str,
        help="Path for de-identified text."
    )
    parser.add_argument(
        "--ref_path",
        required=True,
        type=str,
        help=(
            "The input ref folder with labels"
            " for text files to be deidentified."
        )
    )
    parser.add_argument(
        "--pred_extension",
        default='pred',
        type=str,
        help=("Extension for predictions labels"
              " (default: pred)")
    )
    parser.add_argument(
        "--ref_extension",
        default='gs',
        type=str,
        help=(
            "Extension for gold standard files in"
            " ref folder (default: gs)"
        )
    )
    parser.add_argument(
        '--log',
        type=str,
        default=None,
        help='text file to output false negatives to'
    )

    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='CSV file to output errors for labeling'
    )

    args = parser.parse_args()

    ref_path = Path(args.ref_path)
    pred_path = Path(args.pred_path)

    csv_path = None
    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
        if not os.path.exists(csv_path.parents[0]):
            os.makedirs(csv_path.parents[0])

    log_path = None
    if args.log is not None:
        log_path = Path(args.log)
        if not os.path.exists(log_path.parents[0]):
            os.makedirs(log_path.parents[0])

        log_text = OrderedDict(
            [['False Negatives', ''], ['False Positives', '']]
        )

    input_ext = '.txt'
    gs_ext = args.ref_extension
    if not gs_ext.startswith('.'):
        gs_ext = '.' + gs_ext
    pred_ext = args.pred_extension
    if not pred_ext.startswith('.'):
        pred_ext = '.' + pred_ext

    # read files from folder
    if os.path.exists(args.text_path):
        input_path = Path(args.text_path)
        files = os.listdir(input_path)

        # remove extension and get file list
        input_files = set(
            [f[0:-len(input_ext)] for f in files if f.endswith(input_ext)]
        )

        input_files = sorted(list(input_files))
    else:
        raise ValueError('Input folder %s does not exist.', args.text_path)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(input_files))

    perf_all = {}
    fn_all = []
    fp_all = []
    # keep track of all PHI tokens in this dataset
    for fn in tqdm(input_files, total=len(input_files)):
        # load the text
        with open(input_path / f'{fn}{input_ext}', 'r') as fp:
            text = ''.join(fp.readlines())

        # load output of bert-deid
        fn_pred = pred_path / f'{fn}{pred_ext}'
        df = pd.read_csv(
            fn_pred, header=0, dtype={
                'entity': str,
                'entity_type': str
            }
        )

        # load ground truth
        gs_fn = ref_path / f'{fn}{gs_ext}'
        gs = pd.read_csv(
            gs_fn, header=0, dtype={
                'entity': str,
                'entity_type': str
            }
        )

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

        if (log_path is not None) & (
            (curr_performance['n_token_fp'] > 0) |
            (curr_performance['n_token_fn'] > 0)
        ):
            # print out the false negatives
            indices = OrderedDict(
                [
                    ['False Negatives',
                     np.where(tokens_y & ~tokens_ypred)[0]],
                    ['False Positives',
                     np.where(~tokens_y & tokens_ypred)[0]],
                ]
            )
            for k, idx in indices.items():
                for i in idx:
                    start, stop = tokens[i][1:3]
                    log_text[k] += f'{fn},'
                    log_text[k] += text[max(start -
                                            50, 0):start].replace('\n', ' ')
                    entity = text[start:stop]
                    log_text[k] += "**" + entity.replace('\n', ' ') + "**"
                    log_text[k] += text[stop:min(stop + 50, len(text))].replace(
                        '\n', ' '
                    )
                    log_text[k] += '\n'
                    # write the entry that should be added to the gold-standard
                    if (',' in entity) or ('\n' in entity) or ('"' in entity):
                        entity = '"' + entity.replace('"', '""') + '"'

                    log_text[k] += f'{fn},,{start},{stop},{entity},,\n'

        perf_all[fn] = curr_performance
        fn_all.extend(
            [
                x[0]
                for i, x in enumerate(tokens) if tokens_y[i] & ~tokens_ypred[i]
            ]
        )
        fp_all.extend(
            [
                x[0]
                for i, x in enumerate(tokens) if ~tokens_y[i] & tokens_ypred[i]
            ]
        )

    # convert to dataframe
    df = pd.DataFrame.from_dict(perf_all, orient='index')

    print(df)
    print('\nFalse negatives\n"{}"'.format('","'.join(fn_all)))
    print('\nFalse positives:\n"{}"'.format('","'.join(fp_all)))
    print('\n')

    # summary stats
    se = df['n_token_tp'] / (df['n_token_tp'] + df['n_token_fn'])
    ppv = df['n_token_tp'] / (df['n_token_tp'] + df['n_token_fp'])
    f1 = 2 * se * ppv / (se + ppv)
    print(f'Macro Se: {se.mean():0.3f}')
    print(f'Macro P+: {ppv.mean():0.3f}')
    print(f'Macro F1: {f1.mean():0.3f}')

    se = df['n_token_tp'].sum(
    ) / (df['n_token_tp'].sum() + df['n_token_fn'].sum())
    ppv = df['n_token_tp'].sum(
    ) / (df['n_token_tp'].sum() + df['n_token_fp'].sum())
    f1 = 2 * se * ppv / (se + ppv)
    print(f'Micro Se: {se.mean():0.3f}')
    print(f'Micro P+: {ppv.mean():0.3f}')
    print(f'Micro F1: {f1.mean():0.3f}')

    if log_path is not None:
        # overwrite the log file
        with open(log_path, 'w') as fp:
            for k, text in log_text.items():
                fp.write(f'=== {k} ===\n{text}\n\n')

    if csv_path is not None:
        df.to_csv(csv_path)


if __name__ == "__main__":
    main()
