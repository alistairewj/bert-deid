"""
Runs BERT deid on a set of text files.
Evaluates the output using gold standard annotations.

Optionally outputs mismatches to brat standoff format.
"""
from __future__ import absolute_import, division, print_function

import os
import argparse
import csv
from pathlib import Path
from bisect import bisect_left

from sklearn import metrics
from tqdm import tqdm
import numpy as np
import pandas as pd


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path",
        required=True,
        type=str,
        help=("Folder containing tokens with probabilities.")
    )

    # output files for results
    parser.add_argument(
        '--stats_path',
        type=str,
        default=None,
        help='CSV file to output performance measures'
    )

    args = parser.parse_args()
    return args


def main():
    args = argparser()

    pred_path = Path(args.pred_path)
    pred_files = os.listdir(pred_path)
    tokens = []
    labels = []
    probs = []
    for f in tqdm(pred_files):
        with open(pred_path / f, 'r') as fp:
            csvreader = csv.reader(fp)
            # skip header
            _ = next(csvreader)
            for row in csvreader:
                tokens.append(row[0])
                labels.append(int(row[1]))
                probs.append([float(x) for x in row[2:]])

    # convert probs to numpy array
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    # filter out empty tokens
    idxKeep = [True if len(t.strip()) > 0 else False for t in tokens]

    probs = probs[idxKeep, :]
    labels = labels[idxKeep]
    tokens = [tokens[i] for i, k in enumerate(idxKeep) if k]

    yhat = np.max(probs[:, 1:], axis=1)
    y = (labels > 0).astype(int)

    ppv, se, th = metrics.precision_recall_curve(y, yhat, pos_label=1)

    # reverse arrays
    se = se[-1:0:-1]
    ppv = ppv[-1:0:-1]
    th = th[-1:0:-1]

    # first, the initial sensitivity is whatever gives best balanced error
    scores = []
    for sens in [0.997, 0.99, 0.9827]:
        i = bisect_left(se, sens)
        print(i)
        if i >= len(se):
            i = len(se) - 1
            tt = th[i - 1]
        else:
            tt = th[i]
        f1 = 2 * se[i] * ppv[i] / (se[i] + ppv[i])

        n_tp = np.sum(y[yhat >= tt])
        n_fp = np.sum(1 - y[yhat >= tt])
        n_fn = np.sum(y[yhat < tt])
        n_pos = np.sum(yhat >= tt)
        scores.append([sens, se[i], ppv[i], f1, n_tp, n_fp, n_fn, n_pos])

    scores = pd.DataFrame(scores)
    scores.columns = [
        'Sensitivity', 'Se (Actual)', 'PPV', 'F1', 'TP', 'FP', 'FN', 'n_pos'
    ]

    scores['FN/1000'] = scores['FN'] / len(y) * 1000.0
    scores['FP/1000'] = scores['FP'] / len(y) * 1000.0
    print(
        scores[['Sensitivity', 'PPV', 'F1', 'FN/1000',
                'FP/1000']].set_index('Sensitivity').to_latex()
    )
    print(scores)


if __name__ == "__main__":
    main()
