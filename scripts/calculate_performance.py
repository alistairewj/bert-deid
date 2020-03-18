"""
Runs BERT deid on a set of text files.
Evaluates the output using gold standard annotations.

Optionally outputs mismatches to brat standoff format.
"""
from __future__ import absolute_import, division, print_function

import os
import re
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

from tqdm import tqdm
import pandas as pd
import numpy as np

import stanfordnlp
import spacy
from spacy.lang.en import English

from bert_deid import utils, processors
from bert_deid.tokenization import split_by_pattern
from bert_deid.label import LABEL_MEMBERSHIP, LABEL_SETS, LabelCollection

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path",
        required=True,
        type=str,
        help=(
            "Folder containing prediction sub-folders. "
            "Each subfolder should contain CSV files with predictions."
        )
    )
    parser.add_argument(
        "--text_path", required=True, type=str, help="Path for text."
    )
    parser.add_argument(
        "--ref_path",
        required=True,
        type=str,
        help="Folder with ground truth labels for text files."
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

    # how to group text for evaluation
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='Regex pattern to split text into tokens for eval.'
    )

    # output files for results
    parser.add_argument(
        '--stats_path',
        type=str,
        default=None,
        help='CSV file to output performance measures'
    )
    parser.add_argument(
        '--tokens_path',
        type=str,
        default=None,
        help='CSV file to output tokens with predictions.'
    )

    # label arguments
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        choices=LABEL_SETS,
        help=(
            "The input dataset type. "
            "Valid choices: {}.".format(', '.join(LABEL_SETS))
        ),
    )
    parser.add_argument(
        "--bio",
        action='store_true',
        help="Whether to transform labels to use inside-outside-beginning tags"
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

    args = parser.parse_args()
    return args


def main():
    args = argparser()

    ref_path = Path(args.ref_path)
    pred_path = Path(args.pred_path)

    stats_path = None
    if args.stats_path is not None:
        stats_path = Path(args.stats_path)
        if not os.path.exists(stats_path.parents[0]):
            os.makedirs(stats_path.parents[0])

    tokens_path = None
    if args.tokens_path is not None:
        tokens_path = Path(args.tokens_path)
        if not os.path.exists(tokens_path.parents[0]):
            os.makedirs(tokens_path.parents[0])

        # log_text = OrderedDict(
        #     [['False Negatives', ''], ['False Positives', '']]
        # )

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

    # we will try to evaluate against all sub-folders in the provided pred_path
    # check for sub-folders which have an annotation for all text files
    pred_folders = []
    for f in os.listdir(pred_path):
        pred_subfolder = pred_path / f
        if not pred_subfolder.is_dir():
            continue

        pred_files = os.listdir(pred_subfolder)

        # remove extension and get file list
        pred_files = set(
            [f[0:-len(pred_ext)] for f in pred_files if f.endswith(pred_ext)]
        )
        pred_files = set(pred_files)

        # if we have a pred for all input files, retain this subfolder
        if pred_files.issubset(set(input_files)):
            pred_folders.append(pred_subfolder)

    if len(pred_folders) == 0:
        # we did not find any sub-folders to evaluate - exit fcn
        logger.warning(
            'Did not find subfolders in %d with all predictions', pred_path
        )
        return

    if args.tokenizer == 'stanford':
        # use stanford core NLP to tokenize data
        tokenizer = stanfordnlp.Pipeline(processors='tokenize', lang='en')
    elif args.tokenizer == 'spacy':
        # Create a Tokenizer with the default settings for English
        # including punctuation rules and exceptions
        nlp = spacy.lang.en.English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
    else:
        # will use regex (re.findall) with the given string to create tokens
        tokenizer = args.tokenizer

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(input_files))

    gs = LabelCollection(args.task, args.bio, transform=args.label_transform)
    pred = LabelCollection(args.task, args.bio, transform=args.label_transform)

    comparison = {}
    for field in [
        'model', 'record', 'truth', 'pred', 'token', 'start', 'length'
    ]:
        comparison[field] = []

    perf_all = {}
    # keep track of all PHI tokens in this dataset
    for fn in tqdm(input_files, total=len(input_files)):
        # load the text
        with open(input_path / f'{fn}{input_ext}', 'r') as fp:
            text = ''.join(fp.readlines())

        # load ground truth
        gs_fn = ref_path / f'{fn}{gs_ext}'
        gs.from_csv(gs_fn)

        for pred_folder in pred_folders:
            # load output of bert-deid
            fn_pred = pred_folder / f'{fn}{pred_ext}'
            pred.from_csv(fn_pred)

            text_tar = utils.get_characterwise_labels(gs, text)
            text_pred = utils.get_characterwise_labels(pred, text)

            # now evaluate the character-wise labels using some aggregation
            # an empty tokenizer results in character-wise evaluation
            tokens_true, tokens_pred, tokens, tokens_start, tokens_length = utils.generate_token_arrays(
                text,
                text_tar,
                text_pred,
                tokenizer=tokenizer,
                expand_predictions=True,
                split_true_entities=True
            )

            # retain comparison for later output
            comparison['model'].extend([pred_folder.stem] * len(tokens))
            comparison['record'].extend([fn] * len(tokens))
            comparison['start'].extend(tokens_start)
            comparison['length'].extend(tokens_length)
            comparison['token'].extend(tokens)
            comparison['truth'].extend(
                [gs.id_to_label[t] if t >= 0 else np.nan for t in tokens_true]
            )
            comparison['pred'].extend(
                [gs.id_to_label[t] if t >= 0 else np.nan for t in tokens_pred]
            )

            performance = {}
            performance['model'] = pred_folder.stem
            performance['n_token_phi'] = len(tokens_true)
            performance['n_true_positive'] = sum(
                (tokens_true >= 0) & (tokens_true == tokens_pred)
            )
            performance['n_false_negative'] = sum(
                (tokens_true >= 0) & (tokens_pred == -1)
            )
            performance['n_false_positive'] = sum(
                (tokens_true == -1) & (tokens_pred >= 0)
            )
            performance['n_mismatch'] = sum(
                (tokens_true >= 0) & (tokens_pred >= 0) &
                (tokens_true != tokens_pred)
            )

            perf_all[fn] = performance

    # convert to dataframe
    df = pd.DataFrame.from_dict(perf_all, orient='index')
    df.index.name = 'record'

    print(df.head())

    if stats_path is not None:
        df.to_csv(stats_path)

    comparison = pd.DataFrame.from_dict(comparison, orient='columns')
    comparison.sort_values(['model', 'record', 'start'], inplace=True)
    if tokens_path is not None:
        if tokens_path.suffix == '.gz':
            compression = 'gzip'
        else:
            compression = None
        comparison.to_csv(tokens_path, index=False, compression=compression)


if __name__ == "__main__":
    main()
