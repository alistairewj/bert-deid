from bert_deid import utils
import pandas as pd
import numpy as np
import re
import string
import csv
import glob
import argparse
import logging
from pathlib import Path
import os
import re
from tqdm import tqdm
from collections import OrderedDict

from bert_deid import processors
from bert_deid.label import LABEL_SETS, LABEL_MEMBERSHIP, LabelCollection, LABEL_MAP
from sklearn.metrics import classification_report
import json
"""
Runs BERT deid on a set of text files.
Evaluates the output (matched correct PHI categories) using gold standard annotations.

Optionally outputs mismatches to brat standoff format.
"""

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def find_phi_tokens(token_text, start, end, all_labels):
    """
    Get all phi TOKENS and correponding label, start, end (resolved conflicted-labels token by splitting on conflict)
    """
    n_tokens = 0
    total_tokens = []
    uniques = np.unique(all_labels[start:end])
    if len(uniques) > 1:  # resolve conflicted label for a token
        tokens, starts, ends = utils.split_by_token_entity(
            token_text, all_labels[start:end], start
        )
    else:
        tokens, starts, ends = [token_text], [start], [end]

    for i in range(len(tokens)):
        label = 0
        for j in range(starts[i], ends[i]):
            if all_labels[j] != 0:
                assert (
                    (label != 0 and all_labels[j] == label) or label == 0
                )  # check no more conflicted label for a token
                label = all_labels[j]

        if label != 0:  # a phi
            total_tokens.append((tokens[i], label, starts[i], ends[i]))

        n_tokens += 1
    return n_tokens, total_tokens


def token_info(
    token_text, start, end, true_all_labels, pred_all_labels, id2label_map
):
    """
    Given a token splitted by whitespace with corresponding start and end,
    Return true_label(s), pred_label(s) after resolve conflicted labelling
    """
    true_label, pred_label = [], []
    prev_label_y = true_all_labels[start]
    prev_label_ypred = pred_all_labels[start]
    current_start = start
    for i in range(start + 1, end):
        current_label_y = true_all_labels[i]
        current_label_ypred = pred_all_labels[i]
        if current_label_y != prev_label_y:  # false negative
            # most frequent predicted label for token
            prev_label_ypred = max(
                pred_all_labels[current_start:i],
                key=list(pred_all_labels[current_start:i]).count
            )
            # keep track of PHI tokens
            if prev_label_y != 0:
                true_label.append(id2label_map[prev_label_y])
                pred_label.append(id2label_map[prev_label_ypred])
            prev_label_y = current_label_y
            prev_label_ypred = current_label_ypred
            current_start = i
        elif current_label_ypred != prev_label_ypred:  # false positive
            prev_label_y = max(
                true_all_labels[current_start:i],
                key=list(true_all_labels[current_start:i]).count
            )
            if prev_label_ypred != 0:
                true_label.append(id2label_map[prev_label_y])
                pred_label.append(id2label_map[prev_label_ypred])
            prev_label_y = current_label_y
            prev_label_ypred = current_label_ypred
            current_start = i

    if prev_label_y != 0 or prev_label_ypred != 0:
        true_label.append(id2label_map[prev_label_y])
        pred_label.append(id2label_map[prev_label_ypred])
    assert (len(pred_label) == len(true_label))
    return true_label, pred_label


def token_eval(phi_tokens, all_labels, is_expand):
    """ Compute true positive and false positive when true phi tokens with start and end
     and predicted labels are given.
     Or compute true positive and false negative when pred phi tokens with start and end 
     and true labels are given.
    """
    tp_list = []
    false_list = []
    for token, label, start, end in phi_tokens:
        is_tp = False
        # compute tp: at least one char needs to be true label (if expanding)
        # sum fn: no conlicted label (0 or true label) or (just true label)
        token_label_list = all_labels[start:end]
        if label in token_label_list and sum(
            np.unique(token_label_list) == label
        ):
            if is_expand:
                # if any individual char is flagged, the token is flagged
                tp_list.append((token, label, start, end))
                is_tp = True
            else:
                # need match exact start and end, no punishment on punctuation at start or end
                if token[0] in string.punctuation:
                    token_label_list = token_label_list[1:]
                if token[-1] in string.punctuation:
                    token_label_list = token_label_list[:-1]
                # tp: rest of chars in token must all matched with true label
                if len(np.unique(token_label_list)) == 1:
                    tp_list.append((token, label, start, end))
                    is_tp = True

        if not is_tp:
            false_list.append((token, label, start, end))

    return tp_list, false_list


def merge_BIO_pred(anno, is_binary, is_expand, text):
    sort_by_start = anno.sort_values("start")
    # find starting row index for each PHI predicted
    entity_starts = []
    for i, row in sort_by_start.iterrows():
        if row["entity_type"][0].lower() == "b":
            entity_starts.append(i)
    # create a new dataframe merging BIO annotations to original ann
    new_anno = pd.DataFrame(columns=sort_by_start.columns)
    for i in range(len(entity_starts)):
        current_start_index = entity_starts[i]
        if i == len(entity_starts) - 1:
            next_start_index = len(sort_by_start)
        else:
            next_start_index = entity_starts[i + 1]
        if is_binary:
            entity_type = 'phi'
        else:
            entity_type = sort_by_start["entity_type"].iloc[current_start_index
                                                           ].split("-")[1]
        start = sort_by_start["start"].iloc[current_start_index]
        stop = sort_by_start["stop"].iloc[next_start_index - 1]
        entity = ""
        if is_expand:
            entity += text[start:stop]
        else:
            for j in range(current_start_index, next_start_index - 1):
                entity += str(sort_by_start["entity"].iloc[j])
                # This handles entity evaluation where BERT misses
                # middle of entity but predict start and end correct
                # in such case, token evaluation is worse than entity evaluation
                # if only cares about correctly predict START, STOP, ENTITY TYPE.

                for _ in range(
                    sort_by_start["stop"].iloc[j],
                    sort_by_start["start"].iloc[j + 1]
                ):
                    entity += " "
            entity += str(sort_by_start["entity"].iloc[next_start_index - 1])
        new_anno = new_anno.append(
            {
                "document_id": sort_by_start["document_id"].iloc[0],
                "annotation_id": "",
                "start": start,
                "stop": stop,
                "entity": entity,
                "entity_type": entity_type,
                "cooment": ""
            },
            ignore_index=True
        )

    return new_anno


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path",
        required=True,
        type=str,
        help="Path for predicted albels"
    )

    parser.add_argument(
        "--text_path",
        required=True,
        type=str,
        help="Path for de-identified text"
    )

    parser.add_argument(
        "--ref_path", required=True, type=str, help="Gold standard labels"
    )

    parser.add_argument(
        "--pred_extension",
        default="pred",
        type=str,
        help=("Extension for prediction labels (default: pred)")
    )

    parser.add_argument(
        "--ref_extension",
        default="gs",
        type=str,
        help=("Extension for gold standard files in ref folder (default: gs)")
    )

    parser.add_argument(
        "--binary",
        action="store_true",
        help="Do binary evaluation (PHI or not)"
    )

    parser.add_argument(
        "--expand_eval",
        action="store_true",
        help="Enable more freedom in evalution, as if any individual \
        character is flagged as a phi instance, whole token would be flagged \
            as phi instance"
    )

    parser.add_argument(
        "--data_type",
        default=None,
        type=str,
        required=True,
        choices=LABEL_SETS,
        help="The input dataset type. Valid choices: {}.".format(
            ', '.join(LABEL_SETS)
        ),
    )

    # label transformations
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

    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="text file to output false positive/negative to"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="CSV file to output errors for labelling"
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
            [
                ["False Negatives Token", ''], ['False Negatives Entity', ''],
                ["False Positives Token", ''], ['False Positives Entity', '']
            ]
        )

    input_ext = '.txt'
    gs_ext = args.ref_extension
    if not gs_ext.startswith('.'):
        gs_ext = '.' + gs_ext
    pred_ext = args.pred_extension
    if not pred_ext.startswith('.'):
        pred_ext = '.' + pred_ext

    label_set = LabelCollection(
        data_type=args.data_type, bio=None, transform=args.label_transform
    )

    # get the label to ID map from the label set
    label2id_map = label_set.label_to_id
    id2label_map = {label2id_map[key]: key for key in label2id_map}
    label_list = list(label_set.label_list)
    label_list.remove('O')

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

    is_bio = ''
    if args.bio:
        is_bio += 'with BIO scheme transformation on label'

    if args.binary:
        logger.info("***** Running binary evaluation {} *****".format(is_bio))
    else:
        logger.info(
            "***** Running multi-class evaluation {} *****".format(is_bio)
        )

    logger.info(" Num examples = %d", len(input_files))

    fn_all, fp_all = [], []
    perf_all = {}
    total_eval = 0
    true_labels, pred_labels = [], []
    if args.bio:
        fn_all_entity, fp_all_entity = [], []
    for fn in tqdm(input_files, total=len(input_files)):
        # load the text
        with open(input_path / f'{fn}{input_ext}', 'r') as fp:
            text = ''.join(fp.readlines())

        # load output of bert-deid
        fn_pred = pred_path / f'{fn}{pred_ext}'
        df = pd.read_csv(
            fn_pred,
            header=0,
            delimiter=",",
            dtype={
                'entity': str,
                'entity_type': str
            }
        )
        # load ground truth
        gs_fn = ref_path / f'{fn}{gs_ext}'
        gs = pd.read_csv(
            gs_fn,
            header=0,
            delimiter=",",
            dtype={
                'entity': str,
                'entity_type': str
            }
        )

        # convert start:end PHIs to list of ints representing different PHIs
        text_tar = np.zeros(len(text), dtype=int)
        for i, row in gs.iterrows():
            if args.binary:
                text_tar[row['start']:row['stop']] = 1
            else:
                if args.label_transform is not None:
                    transformed_label = LABEL_MAP[args.label_transform][
                        row['entity_type'].upper()]
                    text_tar[row['start']:row['stop']
                            ] = label2id_map[transformed_label]
                else:
                    text_tar[row['start']:row['stop']] = label2id_map[
                        row['entity_type'].upper()]
        text_pred = np.zeros(len(text), dtype=int)
        for i, row in df.iterrows():
            if args.bio:
                if (
                    'B-' not in row['entity_type'] and
                    'I-' not in row['entity_type']
                ):
                    print(f'type: {row["entity_type"]}')
                    raise ValueError(
                        "Invalid label transform arguments. Data prediction doesn't use BIO scheme."
                    )
                # assert ('b' in row['entity_type'].lower() or 'i' in row['entity_type'])
                entity_type = row['entity_type'][
                    2:]  # remove b- or i-, don't care for char eval
            else:
                if ('B-' in row['entity_type'] or 'I-' in row['entity_type']):
                    raise ValueError(
                        "Required --bio arg. Data prediction uses BIO scheme"
                    )
                entity_type = row['entity_type']
            if args.binary:
                text_pred[row['start']:row['stop']] = 1
            else:
                text_pred[row['start']:row['stop']] = label2id_map[entity_type]

        curr_performance = {}
        curr_performance['n_char'] = len(text)
        curr_performance['n_char_phi'] = len(np.nonzero(text_tar)[0])
        total = 0
        curr_performance['n_char_tp'] = sum(
            1 for e in zip(text_tar, text_pred) if (e[0] == e[1] and e[0] != 0)
        )
        curr_performance['n_char_fp'] = len(np.nonzero(text_pred)[0]
                                           ) - curr_performance['n_char_tp']
        curr_performance['n_char_fn'] = len(np.nonzero(text_tar)[0]
                                           ) - curr_performance['n_char_tp']

        # report performance on token wise
        phi_tokens_true, phi_tokens_pred = [], []
        n_tokens = 0

        # find phi tokens split by whitespace
        # resolve conflicted-label token by splitting on conflict
        for token, start, end in utils.split_by_space(text):
            n_token, token_y = find_phi_tokens(token, start, end, text_tar)
            n_tokens += n_token
            phi_tokens_true.extend(token_y)
            _, token_ypred = find_phi_tokens(token, start, end, text_pred)
            phi_tokens_pred.extend(token_ypred)

            true_label, pred_label = token_info(
                token, start, end, text_tar, text_pred, id2label_map
            )
            true_labels.extend(true_label)
            pred_labels.extend(pred_label)
            assert (len(true_labels) == len(pred_labels))

        tp_list, fn_list = token_eval(
            phi_tokens_true, text_pred, args.expand_eval
        )
        _, fp_list = token_eval(phi_tokens_pred, text_tar, args.expand_eval)

        curr_performance['n_token'] = n_tokens
        curr_performance['n_token_phi'] = len(phi_tokens_true)
        curr_performance['n_token_tp'] = len(tp_list)
        curr_performance['n_token_fp'] = len(fp_list)
        curr_performance['n_token_fn'] = len(fn_list)

        fp_list_entity, fn_list_entity = None, None
        # report performance entity-wise (only when label is transformed)
        if args.bio:
            df = merge_BIO_pred(df, args.binary, args.expand_eval, text)
            if args.binary:
                gs['entity_type'] = 'phi'
            # ignore punctuation punshiment at front/end
            true = utils.ignore_partials(utils.get_entities(gs))
            pred = utils.ignore_partials(utils.get_entities(df))

            tp_list_entity = set(pred) & set(true)
            fp_list_entity = set(pred).difference(tp_list_entity)
            fn_list_entity = set(true).difference(tp_list_entity)

            curr_performance['n_entity_phi'] = len(true)
            curr_performance['n_entity_tp'] = len(tp_list_entity)
            curr_performance['n_entity_fp'] = len(fp_list_entity)
            curr_performance['n_entity_fn'] = len(fn_list_entity)

            fn_all_entity.extend(fn_list_entity)
            fp_all_entity.extend(fp_list_entity)

        if (log_path is not None) & (
            (curr_performance['n_token_fp'] > 0) |
            (curr_performance['n_token_fn'] > 0) | (
                args.bio &
                (fn_list_entity is not None or fp_list_entity is not None)
            )
        ):

            for key in log_text.keys():
                if key == 'False Positives Token':
                    false_list = fp_list
                elif key == 'False Negatives Token':
                    false_list = fn_list
                elif key == 'False Positives Entity' and args.bio:
                    false_list = fp_list_entity
                elif key == 'False Negatives Entity' and args.bio:
                    false_list = fn_list_entity
                else:
                    false_list = []
                # false list: (token, label, start, end)
                sorted_false_list = sorted(false_list, key=lambda x: x[2])
                for (token, label, start, stop) in sorted_false_list:
                    log_text[key] += f'{fn},'
                    log_text[key] += text[max(start -
                                              50, 0):start].replace('\n', ' ')
                    log_text[key] += "**" + token.replace('\n', ' ') + "**"
                    log_text[key] += text[stop:min(stop +
                                                   50, len(text))].replace(
                                                       "\n", " "
                                                   )
                    log_text[key] += "\n"
                    if (',' in token) or ("\n" in token) or ('"' in token):
                        token = '"' + token.replace('"', '""') + '"'
                    label = id2label_map[label]
                    log_text[key] += f'{fn},,{start},{stop},{token},{label},\n'

        perf_all[fn] = curr_performance
        fn_all.extend(fn_list)
        fp_all.extend(fp_list)

    # convert to dataframe
    df = pd.DataFrame.from_dict(perf_all, orient='index')

    print(df)
    # print ("\nFalse negatives:")
    # for fn in fn_all:
    #     print (fn)
    # print ('False positives:')
    # for fp in fp_all:
    #     print (fp)
    # print('\n')

    # summary stats
    se, ppv, f1 = utils.compute_stats(df, True, False)
    print(f'Token Macro Se: {se.mean():0.4f}')
    print(f'Token Macro P+: {ppv.mean():0.4f}')
    print(f'Token Macro F1: {f1.mean():0.4f}')

    se, ppv, f1 = utils.compute_stats(df, True, True)
    print(f'Token Micro Se: {se.mean():0.4f}')
    print(f'Token Micro P+: {ppv.mean():0.4f}')
    print(f'Token Micro F1: {f1.mean():0.4f}')

    if args.bio:
        se, ppv, f1 = utils.compute_stats(df, False, False)
        print(f'Entity Macro Se: {se.mean():0.4f}')
        print(f'Entity Macro P+: {ppv.mean():0.4f}')
        print(f'Entity Macro F1: {f1.mean():0.4f}')

        se, ppv, f1 = utils.compute_stats(df, False, True)
        print(f'Entity Micro Se: {se.mean():0.4f}')
        print(f'Entity Micro P+: {ppv.mean():0.4f}')
        print(f'Entity Micro F1: {f1.mean():0.4f}')

    if log_path is not None:
        # overwrite the log file
        with open(log_path, 'w') as fp:
            for k, text in log_text.items():
                fp.write(f'=== {k} ===\n{text}\n\n')

    if csv_path is not None:
        df.to_csv(csv_path)

    print('length', len(true_labels))
    print('n phi tokens: ', sum(df['n_token_phi']))
    tokens_report = classification_report(
        true_labels, pred_labels, labels=label_list, digits=4
    )
    print("**** token report *****")
    print(tokens_report)
    # with open('/home/jingglin/research/data/pred/i2b2_2014/token_log/test/token_report.json', 'w') as json_file:
    #     json.dumps(tokens_report, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()
