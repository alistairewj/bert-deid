"""Combine BERT results with pydeid results"""
import os
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import pkgutil
from bert_deid.label import PYDEID_FEATURE2LABEL

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

from pydeid.annotators import _patterns
from pydeid.annotators import Pattern
from pydeid.annotation import Document, EntityType
# load all modules on path
pkg = 'pydeid.annotators._patterns'
PATTERN_NAMES = [
    name for _, name, _ in pkgutil.iter_modules(_patterns.__path__)
]
PATTERN_NAMES.remove('_pattern')
_PATTERN_NAMES = PATTERN_NAMES + ['all']


def merge_intervals(intervals):

    sorted_intervals = sorted(intervals, key=lambda tup: tup[0])

    if not sorted_intervals:  # no intervals to merge
        return

    # low and high represent the bounds of the current run of merges
    low, high = sorted_intervals[0]

    for iv in sorted_intervals[1:]:
        if iv[0] <= high:  # new interval overlaps current run
            high = max(high, iv[1])  # merge with the current run
        else:  # current run is over
            yield low, high  # yield accumulated interval
            low, high = iv  # start new run

    yield low, high


def combine_results(bert_df, pydeid_df, text, features, is_bio=False):
    '''
    Post process: combine PHI predicted by BERT and Pydeid into one dataframe
    Args:
        bert_df: DataFrame, containing PHI predicted by BERT
        pydeid_df: DataFrame, containing PHI predicted by pydeid
        text: string, text docs for a file
        features: list of strings, features used by pydeid algorithm
        is_bio: Bool, BIO scheme or not
    '''
    def get_pydeid_intervals(df, feature):
        '''
        Get predicted PHIs with feature as label
        '''
        arr = []
        for i, row in df.iterrows():
            entity_type = row['entity_type'].lower()
            if entity_type == feature:
                arr.append((row['start'], row['stop']))

        return arr

    def get_bert_intervals(df, feature):
        '''
        Get predicted PHIs with label compatible to pydeid feature
        '''
        arr = []
        for i, row in df.iterrows():
            entity_type = row['entity_type'].upper()
            if is_bio:
                entity_type = entity_type[2:]
            if entity_type == PYDEID_FEATURE2LABEL[feature]:
                arr.append((row['start'], row['stop']))

        return arr

    new_df = pd.DataFrame(columns=bert_df.columns)
    if len(bert_df) > 0:
        document_id = bert_df['document_id'].iloc[0]
    elif len(pydeid_df) > 0:
        document_id = pydeid_df['document_id'].iloc[0]
    else:
        return new_df

    for feature in features:
        bert_intervals = get_bert_intervals(bert_df, feature)
        pydeid_intervals = get_pydeid_intervals(pydeid_df, feature)

        combined_intervals = bert_intervals + pydeid_intervals
        combined_intervals = merge_intervals(combined_intervals)

        for start, stop in combined_intervals:
            entity = text[start:stop]
            entity_type = PYDEID_FEATURE2LABEL[feature]
            if is_bio:
                entity_type = 'B-' + entity_type
            new_df = new_df.append(
                {
                    "document_id": document_id,
                    "annotation_id": "",
                    "start": start,
                    "stop": stop,
                    "entity": entity,
                    "entity_type": entity_type,
                    "comment": ""
                },
                ignore_index=True
            )

    # add rest of predicted PHI by BERT model that doesn't match to any feature label
    matched_entity_types = set(
        [PYDEID_FEATURE2LABEL[feature] for feature in features]
    )

    for i, row in bert_df.iterrows():
        entity_type = row['entity_type'].upper()
        if is_bio:
            entity_type = entity_type[2:]
        start, stop = row['start'], row['stop']
        entity = text[start:stop]
        if entity_type not in matched_entity_types:
            if is_bio:
                entity_type = 'B-' + entity_type
            new_df = new_df.append(
                {
                    "document_id": document_id,
                    "annotation_id": "",
                    "start": start,
                    "stop": stop,
                    "entity": entity,
                    "entity_type": entity_type,
                    "comment": ""
                },
                ignore_index=True
            )

    new_df = new_df.sort_values(by='start')

    return new_df


if __name__ == '__main__':
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
        "--bert_result_dir",
        default='/enc_data/pred/bert-i2b2-2014',
        type=str,
        required=True,
        help="BERT annotation result dir.",
    )
    parser.add_argument(
        "--pydeid_result_dir",
        default='/enc_data/pred/pydeid-i2b2-2014',
        type=str,
        required=True,
        help="Pydeid annotation result dir"
    )
    parser.add_argument(
        "--feature",
        type=str,
        nargs='+',
        default=None,
        choices=_PATTERN_NAMES,
        help="rule-based approach with pydeid patterns: "
        f"{', '.join(_PATTERN_NAMES)}"
    )
    parser.add_argument(
        "--bio",
        action='store_true',
        help="Whether to transform labels to use inside-outside-beginning tags"
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Output folder combined result",
    )
    args = parser.parse_args()

    bert_result_path = Path(args.bert_result_dir)
    pydeid_result_path = Path(args.pydeid_result_dir)
    data_path = Path(args.data_dir)

    if args.output_folder is not None:
        output_folder = Path(args.output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        output_header = [
            'document_id', 'annotation_id', 'start', 'stop', 'entity',
            'entity_type', 'comment'
        ]
    else:
        output_folder = None

    bert_result_files = os.listdir(bert_result_path)
    bert_result_files = [f for f in bert_result_files if f.endswith('.pred')]
    data = []

    preds = None
    labels = []

    for fn in tqdm(bert_result_files, total=len(bert_result_files)):

        # load output of bert-deid
        bert_fn_pred = bert_result_path / f'{fn}'
        bert_df = pd.read_csv(
            bert_fn_pred,
            header=0,
            delimiter=",",
            dtype={
                'entity': str,
                'entity_type': str
            }
        )
        pydeid_fn_pred = pydeid_result_path / f'{fn}'
        pydeid_df = pd.read_csv(
            pydeid_fn_pred,
            header=0,
            delimiter=",",
            dtype={
                'entity': str,
                'entity_type': str
            }
        )

        args.patterns = []
        if args.feature is not None:
            for f in args.feature:
                f = f.lower()
                if f not in _PATTERN_NAMES:
                    raise ValueError("Invalid feature name")
                args.patterns.append(f)
        if 'all' in args.patterns:
            args.patterns = PATTERN_NAMES

        with open(data_path / 'txt' / f'{fn[:-4]}txt', 'r') as fp:
            text = ''.join(fp.readlines())

        ex_preds = combine_results(
            bert_df, pydeid_df, text, args.patterns, args.bio
        )
        if output_folder is not None:
            ex_preds.to_csv(output_folder / f'{fn[:-4]}pred', index=False)
