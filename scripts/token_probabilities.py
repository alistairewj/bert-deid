import argparse
import csv
import os
import re
import pickle
from pathlib import Path
from bisect import bisect_left, bisect_right
import logging

import numpy as np
from transformers import BertTokenizer
from bert_deid.model import Transformer
from bert_deid.processors import DeidProcessor
from bert_deid.label import LabelCollection, LABEL_SETS, LABEL_MEMBERSHIP
from bert_deid import utils
from tqdm import tqdm

import spacy
from spacy.lang.en import English
import stanfordnlp

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


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

    return tokens_true, tokens_pred, tokens, tokens_start, tokens_length


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='/enc_data/deid-gs/i2b2_2014/test',
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--model_dir",
        default='/enc_data/models/bert-i2b2-2014',
        type=str,
        help="Path to the model.",
    )
    parser.add_argument(
        "--model_type", default='bert', type=str, help="Type of model"
    )
    parser.add_argument(
        "--task",
        default='i2b2_2014',
        type=str,
        choices=LABEL_SETS,
        help=f"Type of dataset: {', '.join(LABEL_SETS)}.",
    )
    parser.add_argument(
        "--output",
        default='preds.pkl',
        type=str,
        help="Output file",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Output folder for CSV stand-off annotations.",
    )

    # how to group text for evaluation
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='Tokenizer to group text into tokens for eval.'
    )
    args = parser.parse_args()

    if args.tokenizer == 'stanford':
        # use stanford core NLP to tokenize data
        tokenizer = stanfordnlp.Pipeline(processors='tokenize', lang='en')
    elif args.tokenizer == 'spacy':
        # Create a Tokenizer with the default settings for English
        # including punctuation rules and exceptions
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
    else:
        # will use regex (re.findall) with the given string to create tokens
        tokenizer = args.tokenizer

    # load in a trained model
    transformer = Transformer(
        args.model_type, args.model_dir, max_seq_length=128, device='cpu'
    )
    gs = transformer.label_set
    # label_to_id = transformer.label_set.label_to_id

    data_path = Path(args.data_dir)

    if args.output_folder is not None:
        output_folder = Path(args.output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        output_header = ['token', 'label_id'] + list(gs.label_list)
    else:
        output_header = None
        output_folder = None

    files = os.listdir(data_path / 'txt')
    files = [f[0:-4] for f in files if f.endswith('.txt')]

    preds = None

    for f in tqdm(files, total=len(files), desc='Files'):
        with open(data_path / 'txt' / f'{f}.txt', 'r') as fp:
            text = ''.join(fp.readlines())

        f_preds, f_lengths, f_starts = transformer.predict(text)

        if preds is None:
            preds = f_preds
            starts = f_starts
            lengths = f_lengths
        else:
            preds = np.append(preds, f_preds, axis=0)
            starts = np.append(starts, f_starts, axis=0)
            lengths = np.append(lengths, f_lengths, axis=0)

        # load ground truth
        gs_fn = data_path / 'ann' / f'{f}.gs'
        gs.from_csv(gs_fn)
        gs_labels = gs.labels

        # tokenize the text
        tokens = utils.tokenize_text(tokenizer, text)

        # iterate through tokens:
        # get the predicted probability for this token
        #   -> if more than one, favor row with lowest OTHER probability (favor classifying as positive)
        #   -> if none, classify as object

        probs = []
        start = 0
        g = 0
        for token in tokens:
            # sometimes we have empty tokens on their own
            if len(token) == 0:
                continue

            # find where the evaluation token starts
            start = text.find(token, start)
            stop = start + len(token)

            # find where our model labeled token matches
            i = bisect_left(f_starts, start)
            j = bisect_left(f_starts, stop)

            if j > i:
                # select the label which is *least* likely to be 'O'
                idx = np.argmin(f_preds[i:j, gs.label_to_id['O']], axis=0)
                prob = f_preds[i + idx, :]
            else:
                # no probabilities to agg
                # output very high probability for 'OTHER' category
                prob = -100 * np.ones(len(gs.label_list))
                prob[gs.label_to_id['O']] = 100

            # default label_id is "OTHER"
            # this will be used if:
            #   (1) no longer have any gold truth labels (g >= len(gs_labels))
            #   (2) have yet to reach the next label (stop <= gs_label[g].start)
            label_id = gs.label_to_id['O']
            # get the label for this token
            while g < len(gs_labels):
                if stop <= gs_labels[g].start:
                    # token is before current ground truth label
                    break

                if start >= (gs_labels[g].start + gs_labels):
                    # skip to next label as we have passed this one
                    g += 1
                else:
                    label_id = gs.label_to_id[gs_labels[g].entity_type]
                    break

            # now that we have a probability for this token
            # append it to our probs
            probs.append([token, label_id] + prob.tolist())

            # token_true = text_tar[start:start + len(token)]
            # token_pred = text_pred[start:start + len(token)]

        # output these probabilities to file
        if output_folder is not None:
            with open(output_folder / f'{f}.prob', 'w') as fp:
                csvwriter = csv.writer(fp)
                csvwriter.writerow(output_header)
                csvwriter.writerows(probs)


if __name__ == '__main__':
    main()