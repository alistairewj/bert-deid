"""Compare results to gold annotation"""
import os
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import pkgutil
import pydeid
import numpy as np
import re

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import bert_deid.utils as utils
from bert_deid import processors
from bert_deid.label import LabelCollection, LABEL_SETS, LABEL_MEMBERSHIP, LABEL_MAP

from pydeid.annotators import _patterns
# load all modules on path
pkg = 'pydeid.annotators._patterns'
PATTERN_NAMES = [
    name for _, name, _ in pkgutil.iter_modules(_patterns.__path__)
]
PATTERN_NAMES.remove('_pattern')
_PATTERN_NAMES = PATTERN_NAMES + ['all']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='/enc_data/deid-gs/i2b2_2014/test/',
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--text_path",
        default='/enc_data/deid-gs/i2b2_2014/test/txt',
        type=str,
        required=True,
        help="The input text dir.",
    )
    parser.add_argument(
        "--ref_path",
        default='/enc_data/deid-gs/i2b2_2014/test/ann',
        type=str,
        required=True,
        help="The input annotation dir.",
    )
    parser.add_argument(
        "--pred_path",
        default='/enc_data/data/pred/test',
        type=str,
        required=True,
        help="The prediction dir.",
    )
    parser.add_argument(
        "--data_type", default=None, type=str, required=True, help="data type"
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help=
        "Output folder for PHI in each file that are not predicted by model",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        type=str,
        help="Output csv for overall stats",
    )
    parser.add_argument(
        "--gs_ext",
        default='.gs',
        type=str,
        help="Gold annotation extension",
    )
    parser.add_argument(
        "--pred_ext",
        default='.pred',
        type=str,
        help="Pred annotation extension",
    )
    parser.add_argument(
        "--input_ext",
        default='.txt',
        type=str,
        help="Input text extension",
    )
    _LABEL_TRANSFORMS = list(LABEL_MEMBERSHIP.keys())
    parser.add_argument(
        "--label_transform",
        default=None,
        choices=_LABEL_TRANSFORMS,
        help=(
            "Map labels using pre-coded transforms: "
            f"{', '.join(_LABEL_TRANSFORMS)}"
        )
    )
    parser.add_argument(
        "--bio",
        action='store_true',
        help="Whether to transform labels to use inside-outside-beginning tags"
    )
    args = parser.parse_args()

    ref_path = Path(args.ref_path)
    pred_path = Path(args.pred_path)

    if args.output_folder is not None:
        output_folder = Path(args.output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        output_header = ['document_id', 'start', 'stop', 'token', 'token_type']
    else:
        output_folder = None

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        if not os.path.exists(output_csv.parents[0]):
            os.makedirs(output_csv.parents[0])

    # read files from folder
    if os.path.exists(args.text_path):
        input_path = Path(args.text_path)
        files = os.listdir(input_path)

        # remove extension and get file list
        input_files = set(
            [
                f[0:-len(args.input_ext)]
                for f in files if f.endswith(args.input_ext)
            ]
        )

        input_files = sorted(list(input_files))
    else:
        raise ValueError('Input folder %s does not exist.', args.text_path)

    label_set = LabelCollection(
        args.data_type, bio=args.bio, transform=args.label_transform
    )
    processor = processors.DeidProcessor(args.data_dir, label_set)

    labels = processor.label_set.label_list
    label2id = processor.label_set.label_to_id
    # invert the mapping from label to ID
    id2label = {i: label for label, i in label2id.items()}

    df_all = pd.DataFrame(
        columns=['document_id', 'n_token', 'n_token_phi'] + list(labels[1:]) +
        ['total_tp', 'total_fp', 'total_fn', 're', 'p+', 'F1']
    )
    # keep track of all PHI tokens in this dataset
    for fn in tqdm(input_files, total=len(input_files)):
        # load the text
        with open(input_path / f'{fn}{args.input_ext}', 'r') as fp:
            text = ''.join(fp.readlines())

        # load output of bert-deid
        fn_pred = pred_path / f'{fn}{args.pred_ext}'
        df = pd.read_csv(
            fn_pred, header=0, dtype={
                'entity': str,
                'entity_type': str
            }
        )

        # load ground truth
        gs_fn = ref_path / f'{fn}{args.gs_ext}'
        gs = pd.read_csv(
            gs_fn, header=0, dtype={
                'entity': str,
                'entity_type': str
            }
        )

        # binary vector indicating PHI/not phi
        text_tar = np.zeros(len(text))

        # map (start, stop): row index in gs
        for i, row in gs.iterrows():
            entity_type = row['entity_type'].upper()
            if args.label_transform is not None:
                entity_type = LABEL_MAP[args.label_transform][entity_type]

            text_tar[row['start']:row['stop']] = label2id[entity_type]

        # binary vector indicating if predicted as phi
        text_pred = np.zeros(len(text), dtype=bool)
        for i, row in df.iterrows():
            text_pred[row['start']:row['stop']] = True

        # report performance token wise
        curr_fn = pd.DataFrame(
            columns=['document_id', 'start', 'stop', 'token', 'token_type']
        )

        pattern = re.compile(r'\s')
        n_tokens = 0
        n_tokens_phi = 0
        n_tokens_fp, n_tokens_fn, n_tokens_tp = 0, 0, 0
        for token, start, end in utils.pattern_spans(text, pattern):
            n_tokens += 1
            token_tar = False
            token_pred = False

            # if any of the individual characters are flagged
            # .. then we consider the whole token to be flagged
            for i in range(start, end):
                # not an object
                if text_tar[i] != 0:
                    token_tar = text_tar[i]
                if text_pred[i]:
                    token_pred = True

            if token_tar != 0:
                n_tokens_phi += 1

            # only get detailed stats for false negative here
            if token_tar != 0 and not token_pred:
                n_tokens_fn += 1
                curr_fn = curr_fn.append(
                    {
                        'document_id': fn,
                        'start': start,
                        'stop': end,
                        'token': token,
                        'token_type': id2label[token_tar]
                    },
                    ignore_index=True
                )

            elif token_tar == 0 and token_pred:
                n_tokens_fp += 1

            elif token_tar != 0 and token_pred:
                n_tokens_tp += 1

        if output_folder is not None:
            curr_fn.to_csv(output_folder / f'{fn}.csv', index=False)

        df_current = {label: 0 for label in labels[1:]}
        df_current['document_id'] = fn
        for i, row in curr_fn.iterrows():
            label = row['token_type']
            df_current[label] += 1
        df_current['n_token'] = n_tokens
        df_current['n_token_phi'] = n_tokens_phi
        df_current['total_fn'] = n_tokens_fn
        df_current['total_fp'] = n_tokens_fp
        df_current['total_tp'] = n_tokens_tp
        if n_tokens_tp + n_tokens_fn == 0:
            df_current['re'] = 1.0
        else:
            df_current['re'] = round(
                n_tokens_tp / (n_tokens_tp + n_tokens_fn), 4
            )
        if n_tokens_tp + n_tokens_fp == 0:
            df_current['p+'] = 1.0
        else:
            df_current['p+'] = round(
                n_tokens_tp / (n_tokens_tp + n_tokens_fp), 4
            )

        if df_current['re'] + df_current['p+'] == 0:
            df_current['F1'] = 0.0
        else:
            df_current['F1'] = round(
                2 * (df_current['re'] * df_current['p+']) /
                (df_current['re'] + df_current['p+']), 4
            )
        df_all = df_all.append(df_current, ignore_index=True)

    df_total_stats = {'document_id': 'All'}
    for label in labels[1:]:
        df_total_stats[label] = sum(df_all[label])
    df_total_stats['n_token'] = sum(df_all['n_token'])
    df_total_stats['n_token_phi'] = sum(df_all['n_token_phi'])
    df_total_stats['total_fn'] = sum(df_all['total_fn'])
    df_total_stats['total_fp'] = sum(df_all['total_fp'])
    df_total_stats['total_tp'] = sum(df_all['total_tp'])
    df_total_stats['re'] = round(
        df_total_stats['total_tp'] /
        (df_total_stats['total_tp'] + df_total_stats['total_fn']), 4
    )
    df_total_stats['p+'] = round(
        df_total_stats['total_tp'] /
        (df_total_stats['total_tp'] + df_total_stats['total_fp']), 4
    )
    df_total_stats['F1'] = round(
        2 * df_total_stats['re'] * df_total_stats['p+'] /
        (df_total_stats['re'] + df_total_stats['p+']), 4
    )
    df_all = df_all.append(df_total_stats, ignore_index=True)

    if output_csv is not None:
        df_all.to_csv(output_csv, index=False)
