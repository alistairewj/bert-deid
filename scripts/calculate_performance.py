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


def get_characterwise_labels(label_set, text):
    """
    Given a label collections, outputs an integer vector with the label_id.
    
    Integer vectors are the same length as the text.
    """
    # integer vector indicating truth
    label_ids = -1 * np.ones(len(text), dtype=int)
    for label in label_set.labels:
        label_ids[label.start:label.start +
                  label.length] = label_set.label_to_id[label.entity_type]

    return label_ids


def split_iterator(pattern, text):
    """
    Iterator that splits text using a regex pattern.
    
    Returns
    -------
    token, start, stop
        Tuple containing the token, the start index of the token
        in the original string, and the end index of the
    """

    tokens = pattern.split(text)

    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)


def split_with_offsets(pattern, text):
    """
    Function that wraps the pattern span iterator.
    """
    tokens_with_spans = list()
    for token, start, end in split_iterator(pattern, text):
        tokens_with_spans.append([start, end, token])

    return tokens_with_spans


def mode(values, ignore_value=None):
    """Get the most frequent value, ignoring a specified value if desired."""
    if len(values) == 0:
        raise ValueError('Cannot calculate mode of length 0 array.')

    p_unique, p_counts = np.unique(values, return_counts=True)
    # remove our ignore index
    if ignore_value is not None:
        idx = np.where(p_unique == ignore_value)[0]
        if len(idx) > 0:
            # we know p_unique is unique, so delete the only element of idx
            p_unique = np.delete(p_unique, idx[0])
            p_counts = np.delete(p_counts, idx[0])

    return p_unique[np.argmax(p_counts)]


def expand_id_to_token(token_pred, ignore_value=None):
    # get most frequent label_id for this token
    p_unique, p_counts = np.unique(token_pred, return_counts=True)

    if len(p_unique) <= 1:
        return token_pred

    # remove our ignore index
    if ignore_value is not None:
        idx = np.where(p_unique == ignore_value)[0]
        if len(idx) > 0:
            # we know p_unique is unique, so get the only element
            p_unique = np.delete(p_unique, idx[0])
            p_counts = np.delete(p_counts, idx[0])

    if len(p_unique) == 1:
        idx = 0
    else:
        # TODO: warn user if we broke a tie by taking lowest ID
        idx = np.argmax(p_counts)

    # re-create the array with only the most frequent label
    token_pred = np.ones(len(token_pred), dtype=int) * p_unique[idx]
    return token_pred


def generate_token_arrays(
    text,
    text_tar,
    text_pred,
    tokenizer=None,
    expand_predictions=True,
    split_true_entities=True,
    ignore_value=None
):
    """
    Evaluate performance of prediction labels compared to ground truth.


    Args
        text_tar - N length numpy array with integers for ground truth labels
        text_pred - N length numpy array with integers for predicted labels
        tokenizer - Determines the granularity level of the evaluation.
            None or '' - character-wise evaluation
            r'\w' - word-wise evaluation
        expand_predictions - If a prediction is partially made for a
            token, expand it to cover the entire token. If not performed,
            then partially labeled tokens are treated as missed detections.
        split_true_entities - The ground truth label for a single token
            may correspond to two distinct classes (e.g. if word splitting,
            John/2010 would be one token but have two ground truth labels).
            Enabling this argument splits these tokens.
        ignore_value - Ignore a label_id in the evaluation. Useful for ignoring
            the 'other' category.
    """
    # split text for token evaluation
    if isinstance(tokenizer, stanfordnlp.pipeline.core.Pipeline):
        doc = tokenizer(text)
        # extract tokens from the parsed text
        tokens_base = [
            token.text for sentence in doc.sentences
            for token in sentence.tokens
        ]
    elif isinstance(tokenizer, spacy.tokenizer.Tokenizer):
        doc = tokenizer(text)
        # extract tokens from the parsed text
        tokens_base = [token.text for token in doc]
    else:
        if tokenizer is None:
            tokenizer = ''
        # treat string as a regex
        tokens_base = re.findall(tokenizer, text)

    tokens = []
    tokens_pred = []
    tokens_true = []
    tokens_start, tokens_length = [], []

    n_tokens = 0

    start = 0
    for token in tokens_base:
        # sometimes we have empty tokens on their own
        if len(token) == 0:
            continue
        start = text.find(token, start)
        token_true = text_tar[start:start + len(token)]
        token_pred = text_pred[start:start + len(token)]

        if all(token_true == -1) & all(token_pred == -1):
            # skip tokens which are not labeled
            start += len(token)
            n_tokens += 1
            continue

        if split_true_entities:
            # split the single token into subtokens, based on the true entity
            idxDiff = np.diff(token_true, prepend=0)
            if any(idxDiff > 0):
                # split
                idxDiff = np.diff(token_true, prepend=0)
                subtok_start = 0
                subtoken_true, subtoken_pred = [], []
                for subtok_end in np.where(idxDiff > 0)[0]:
                    subtoken_true.append(token_true[subtok_start:subtok_end])
                    subtoken_pred.append(token_pred[subtok_start:subtok_end])
                    subtok_start = subtok_end
                if subtok_end < len(token_true):
                    # add final token
                    subtoken_true.append(token_true[subtok_start:])
                    subtoken_pred.append(token_pred[subtok_start:])
            else:
                # in this case, there is only 1 label_id for the entire token
                # so we can just wrap in a list for the iterator later
                subtoken_true = [token_true]
                subtoken_pred = [token_pred]
        else:
            # do not split a token if there is more than 1 ground truth
            # consequently, tokens with multiple labels will be treated
            # as equal to the most frequent label
            subtoken_true = [token_true]
            subtoken_pred = [token_pred]

        # now iterate through our sub-tokens
        # often this is a length 1 iterator
        for token_true, token_pred in zip(subtoken_true, subtoken_pred):
            if len(token_true) == 0:
                continue

            if expand_predictions:
                # expand the most frequent ID to cover the entire token
                token_pred = expand_id_to_token(token_pred, ignore_value=-1)
                token_true = expand_id_to_token(token_true, ignore_value=-1)

            # get the length of the token for later
            token_len = len(token_true)

            # aggregate IDs for this token into the most frequent value
            if len(token_true) == 0:
                token_true = -1
            else:
                token_true = mode(token_true, ignore_value)
            if len(token_pred) == 0:
                token_pred = -1
            else:
                token_pred = mode(token_pred, ignore_value)

            # append the prediction for this token
            tokens_true.append(token_true)
            tokens_pred.append(token_pred)
            tokens.append(text[start:start + token_len])
            tokens_start.append(start)
            tokens_length.append(token_len)

            start += token_len
            # keep track of total tokens assessed
            n_tokens += 1

    # now we have a list of tokens with preds
    tokens_true = np.asarray(tokens_true, dtype=int)
    tokens_pred = np.asarray(tokens_pred, dtype=int)

    return tokens_true, tokens_pred, tokens, tokens_start, tokens_length, n_tokens


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

            text_tar = get_characterwise_labels(gs, text)
            text_pred = get_characterwise_labels(pred, text)

            # now evaluate the character-wise labels using some aggregation
            # an empty tokenizer results in character-wise evaluation
            tokens_true, tokens_pred, tokens, tokens_start, tokens_length, n_tokens = generate_token_arrays(
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
            performance['n_tokens'] = n_tokens
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
