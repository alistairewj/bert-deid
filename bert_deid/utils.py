from __future__ import absolute_import, division, print_function

import json
import os
import logging
import csv
import argparse
import warnings
import re

from sympy import Interval, Union
from bert_deid import model
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig
from tqdm import tqdm
import torch
import pandas as pd


def harmonize_label(label, grouping='i2b2'):
    """
    Groups entities into one of the i2b2 2014 categories. Each dataset has a
    slightly different set of entities. This harmonizes them.

    Args:
        label (str): label to be harmonized (e.g. 'USERNAME').

    Returns:
        harmonized (str): the harmonized label (e.g. 'NAME')..
    """

    # default harmonization is into i2b2 form
    labels = [
        ['NAME', ['NAME', 'DOCTOR', 'PATIENT', 'USERNAME', 'HCPNAME',
                  'RELATIVEPROXYNAME', 'PTNAME', 'PTNAMEINITIAL']],
        ['PROFESSION', ['PROFESSION']],
        ['LOCATION', ['LOCATION', 'HOSPITAL', 'ORGANIZATION',
                      'STREET', 'STATE',
                      'CITY', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
                      'PROTECTED_ENTITY', 'PROTECTED ENTITY',
                      'NATIONALITY']],
        ['AGE', ['AGE', 'AGE_>_89', 'AGE > 89']],
        ['DATE', ['DATE', 'DATEYEAR']],
        ['ID', ['BIOID', 'DEVICE', 'HEALTHPLAN',
                'IDNUM', 'MEDICALRECORD', 'ID', 'OTHER']],
        ['CONTACT', ['EMAIL', 'FAX', 'PHONE', 'URL',
                     'IPADDR', 'IPADDRESS', 'CONTACT']]
    ]

    if grouping == 'i2b2':
        pass
    elif grouping == 'hipaa':
        labels = [
            ['NAME', ['NAME', 'PATIENT', 'USERNAME']],
            ['LOCATION', ['LOCATION', 'ORGANIZATION',
                          'STREET', 'CITY', 'ZIP', ]],
            ['AGE', ['AGE', 'AGE_>_89', 'AGE > 89']],
            ['DATE', ['DATE', 'DATEYEAR']],
            ['ID', ['BIOID', 'DEVICE', 'HEALTHPLAN',
                    'IDNUM', 'MEDICALRECORD', 'ID', 'OTHER']],
            ['CONTACT', ['EMAIL', 'FAX', 'PHONE', 'CONTACT']],
            ['O', [
                'DOCTOR', 'HCPNAME',
                'PROFESSION', 'HOSPITAL',
                'ORGANIZATION', 'STATE', 'COUNTRY',
                'LOCATION-OTHER',
                'PROTECTED_ENTITY', 'PROTECTED ENTITY',
                'NATIONALITY',
                # it is unclear whether these are HIPAA in i2b2 paper
                'URL', 'IPADDR', 'IPADDRESS'
            ]]
        ]
    else:
        raise ValueError(f'Unrecognized grouping {grouping}')

    label_to_type = {}
    for label_map in labels:
        for l in label_map[1]:
            label_to_type[l] = label_map[0]

    return label_to_type[label.upper()]


def combine_entity_types(df, lowercase=True):
    """
    Merge together similar entity types.

    Inputs:
    df (dataframe): annotation dataframe
    lowercase (boolean): output all entities as lowercase.
    """

    # coalesce pydeid types to types understood by the annotation tool
    # mainly this converts all name related labels to "Name"
    type_dict = {'initials': 'Name',
                 'firstname': 'Name',
                 'lastname': 'Name'}
    df['entity_type'] = df['entity_type'].apply(
        lambda x: type_dict[x] if x in type_dict else x)

    # use the annotator name to infer the entity type
    type_fix = {'wmrn': 'Identifier',
                'name': 'Name',
                'initials': 'Name'}
    for typ in type_fix:
        if 'annotator' in df.columns:
            idx = df['entity_type'].isnull() & \
                (df['annotator'].str.contains(typ))
            if idx.any():
                df.loc[idx, 'entity_type'] = type_fix[typ]

    df['entity_type'].fillna('Other', inplace=True)

    # lowercase all entities
    if lowercase:
        df['entity_type'] = df['entity_type'].str.lower()

    return df


def adjust_interval(interval):
    """
    Ensures the interval is left closed, right open.
    This is consistent with python indexing.
    """
    if interval.left_open:
        interval = Interval(
            interval.left + 1, interval.right)
    if not interval.right_open:
        interval = Interval(
            interval.left, interval.right + 1, False, True)
    return interval


def merge_intervals(df, dist=0, text=None):
    """Merge intervals if they are `dist` characters apart.

    Setting `dist` == 0 merges only overlapping intervals.

    Args:
        df (pd.DataFrame): Dataframe with intervals to be merged.
        dist (int): The maximum distance between the end of one
            interval to the start of the next required in order
            to merge them.
        text (str): If provided, entities are updated using text.
            If text is not provided, and dist >= 1, ? characters
            are added as the dataframe does not contain the true
            character.

    Returns:
        pd.DataFrame: The dataframe with merged intervals.
    """
    if df.shape[0] == 0:
        return df

    df.sort_values(['document_id', 'entity_type',
                    'start', 'stop'], inplace=True)
    idx_merged = list()
    added_questions = set()

    for (_, entity_type), grp in df.groupby(['document_id', 'entity_type']):
        n = 0

        for idx, row in grp.iterrows():
            if n == 0:
                idx_merged.append(idx)
                start, stop = row['start'], row['stop']
                entity = row['entity']
                n += 1
                continue

            # check if we should merge
            if (row['start'] - dist) <= stop:
                # start of this interval is close to previous stop
                # -> we are merging intervals

                # add the additional parts of the entity
                update_stop = stop - row['start']

                # add in ? if we have unknown characters
                if update_stop < 0:
                    added_questions.add(idx_merged[-1])
                    entity += '?'*-update_stop + row['entity']
                else:
                    entity += row['entity'][update_stop:]

                # merge the intervals by updating stop
                stop = row['stop']

                n += 1
            else:
                # finished collecting overlapping intervals
                # update the dataframe at the index of the start of this chunk
                df.loc[idx_merged[-1], 'start'] = start
                df.loc[idx_merged[-1], 'stop'] = stop
                df.loc[idx_merged[-1], 'entity_type'] = entity_type
                df.loc[idx_merged[-1], 'entity'] = entity

                # start the new interval to be merged
                start, stop = row['start'], row['stop']
                entity = row['entity']
                idx_merged.append(idx)
                n += 1

        # update the dataframe with the final interval
        df.loc[idx_merged[-1], 'start'] = start
        df.loc[idx_merged[-1], 'stop'] = stop
        df.loc[idx_merged[-1], 'entity_type'] = entity_type
        df.loc[idx_merged[-1], 'entity'] = entity

    # subselect to the merged intervals
    df = df.loc[idx_merged, :]

    if len(added_questions) > 0:
        if text is None:
            # warn the user that we added ?s for unknown characters
            warnings.warn('Added ? characters to entities for unknown characters',
                          category=RuntimeWarning)
        else:
            # update entities with true characters
            for idx in added_questions:
                start, stop = df.loc[idx, 'start'], df.loc[idx, 'stop']
                df.loc[idx, 'entity'] = text[start:stop]

    return df


def simplify_bert_ann(anns, text, lowercase=True, dist=1):
    """
    Simplify the annotations output by BERT.

    Inputs:
    anns (dataframe): annotation dataframe
    text (str): text annotated by the dataframe
    """
    if anns.shape[0] == 0:
        return anns

    # lower case entity types and collapse similar entities
    anns = combine_entity_types(anns, lowercase=lowercase)

    # combine nearby entities into a single group
    anns = merge_intervals(anns, dist=dist, text=text)

    # output new dataframe
    return anns


def suppress_partials(txt, span, start=None, stop=None):
    """
    When outputting anns, ignore partial misses if they are special chars.
    e.g. we don't mind if our ann method misses the period of the sentence.
    """
    if start is None:
        start = 0
    if stop is None:
        stop = len(txt)

    span_start, span_end = span.split(' ')
    span_start, span_end = int(span_start), int(span_end)
    b = span_start - start
    e = len(txt) - (stop - span_end)
    txt_to_check = txt[b:e]
    # suppress spaces as partial misses
    if txt_to_check in (' ', '\r', '\n', ';', ':', '-'):
        return True, txt_to_check

    return False, txt_to_check


def compare(goldstandard, comparison):
    """
    Report on what proportion of the gold standard corpus is
    covered by the given annotations.

    Evaluation
    - exact - 0/1 an exact match on interval (1)
    - partial - 0/1 partially detected (counts as miss)

    False positives
    - proportion of string incorrectly labelled as PHI (excluding white space)
    - or proportion of labelled text beyond true labels
    """

    group_names = ['document_id']
    cols = ['document_id', 'annotation_id', 'start', 'stop', 'entity']

    # stack gold standard and annotations into the same dataframe
    comparison['annotation_id'] = comparison['annotator']
    df = pd.concat([goldstandard[cols], comparison[cols]],
                   ignore_index=True, axis=0)

    # delineate gold standard from annotation
    df['source'] = 'gs'
    df.loc[goldstandard.shape[0]:, 'source'] = 'ann'

    # performance is list of lists
    # index, document_id, exact, partial, missed, start, stop
    # if partial, start/stop denote the missed section
    # otherwise, they encompass the entire entity
    performance = list()

    n_groups = df[group_names].drop_duplicates().shape[0]
    # iterate through each document
    for grp_idx, grp in tqdm(df.groupby(group_names), total=n_groups):
        idxG = grp['source'] == 'gs'
        # create right-open intervals
        cmp_intervals = [[x[0], x[1], False, True]
                         for x in grp.loc[~idxG, ['start', 'stop']].values]
        cmp_intervals = [Interval(*c) for c in cmp_intervals]
        cmp_intervals = Union(*cmp_intervals)

        for idx, row in grp.loc[idxG, :].iterrows():
            # indices are right open
            i = Interval(row['start'], row['stop'], False, True)
            overlap = i.intersect(cmp_intervals)
            mismatch = i - overlap
            # exact match
            if mismatch.is_EmptySet:
                span = '{} {}'.format(row['start'], row['stop'])
                performance.append(
                    [grp_idx, row['annotation_id'], 1, 0, 0, span])
            # partial match
            else:
                # no match
                if mismatch == i:
                    span = '{} {}'.format(row['start'], row['stop'])
                    performance.append(
                        [grp_idx, row['annotation_id'], 0, 0, 1, span])
                else:
                    if type(mismatch) is Union:
                        # we have non-continuous segments in our mismatch
                        span = []
                        for m in mismatch.args:
                            m = adjust_interval(m)
                            span.append('{} {}'.format(m.left, m.right))
                        # brat format: non-contiguous segments are delimited by ';'
                        span = ';'.join(span)
                    else:
                        mismatch = adjust_interval(mismatch)
                        span = '{} {}'.format(mismatch.left, mismatch.right)

                    performance.append(
                        [grp_idx, row['annotation_id'], 0, 1, 0, span])

    # convert back to a dataframe with same index as gs
    performance = pd.DataFrame.from_records(
        performance,
        columns=['document_id', 'annotation_id',
                 'exact', 'partial', 'missed',
                 'span']
    )
    return performance


def compare_single_doc(gs, ann):
    """
    Report on what proportion of a gold standard dataframe is
    covered by the given annotations.
    Assumes all annotations correspond to the same single document.

    Evaluation
    - exact - 0/1 an exact match on interval (1)
    - partial - 0/1 partially detected (counts as miss)

    False positives
    - proportion of string incorrectly labelled as PHI (excluding white space)
    - or proportion of labelled text beyond true labels
    """

    # short circuit comparisons if trivial cases
    if (gs.shape[0] == 0) | (ann.shape[0] == 0):
        # if both df are empty, we will output an empty dataframe
        performance = []
        if gs.shape[0] > 0:
            doc_id = gs['document_id'].values[0]
            # append gold standard rows as misses
            for i, row in gs.iterrows():
                span = '{} {}'.format(row['start'], row['stop'])
                performance.append(
                    [doc_id, row['annotation_id'], 0, 0, 1, span])

        # if ann.shape[0] == 0:
        #     # append ann rows as false positives
        #     for i, row in ann.iterrows():
        #         span = '{} {}'.format(row['start'], row['stop'])
        #         performance.append(
        #             [doc_id, row['annotation_id'], 0, 0, 0, span])

        performance = pd.DataFrame.from_records(
            performance,
            columns=['document_id', 'annotation_id',
                     'exact', 'partial', 'missed',
                     'span']
        )
        return performance

    # performance is list of lists
    # index, document_id, exact, partial, missed, start, stop
    # if partial, start/stop denote the missed section
    # otherwise, they encompass the entire entity
    performance = list()
    if gs.shape[0] > 0:
        doc_id = gs['document_id'].values[0]
    else:
        doc_id = ann['document_id'].values[0]

    # create right-open intervals
    cmp_intervals = [[x[0], x[1], False, True]
                     for x in ann.loc[:, ['start', 'stop']].values]
    cmp_intervals = [Interval(*c) for c in cmp_intervals]
    cmp_intervals = Union(*cmp_intervals)

    for _, row in gs.iterrows():
        # indices are right open
        i = Interval(row['start'], row['stop'], False, True)
        overlap = i.intersect(cmp_intervals)
        mismatch = i - overlap
        # exact match
        if mismatch.is_EmptySet:
            span = '{} {}'.format(row['start'], row['stop'])
            performance.append(
                [doc_id, row['annotation_id'], 1, 0, 0, span])
        # partial match
        else:
            # no match
            if mismatch == i:
                span = '{} {}'.format(row['start'], row['stop'])
                performance.append(
                    [doc_id, row['annotation_id'], 0, 0, 1, span])
            else:
                if type(mismatch) is Union:
                    # we have non-continuous segments in our mismatch
                    span = []
                    for m in mismatch.args:
                        m = adjust_interval(m)
                        span.append('{} {}'.format(m.left, m.right))
                    # brat format: non-contiguous segments are delimited by ';'
                    span = ';'.join(span)
                else:
                    mismatch = adjust_interval(mismatch)
                    span = '{} {}'.format(mismatch.left, mismatch.right)

                performance.append(
                    [doc_id, row['annotation_id'], 0, 1, 0, span])

    # convert back to a dataframe with same index as gs
    performance = pd.DataFrame.from_records(
        performance,
        columns=['document_id', 'annotation_id',
                 'exact', 'partial', 'missed',
                 'span']
    )
    return performance


def output_to_brat(doc_id, df, path, text=None):
    with open(os.path.join(path, f'{doc_id}.ann'), 'w') as fp:
        partial_matches = list()
        missed = list()

        # if annotation ids are well behaved, keep them
        if df['annotation_id'].isnull().any():
            reset_annotations = True
        else:
            idx = df['annotation_id'].str.extract(r'(T[0-9]+)').values
            idx = [idx[i] != df['annotation_id'].values[i]
                   for i in range(df.shape[0])]
            reset_annotations = any(idx)

        if reset_annotations:
            # if any nulls, then reset the annotation IDs
            df['annotation_id'] = df.index
            df['annotation_id'] = df['annotation_id'].map(lambda x: f'T{x}')

        for idx, row in df.iterrows():
            a = row['annotation_id']

            # if entity available, use it
            entity = ''
            if text is not None:
                # we prefer to rederive entity from the text
                for s in row['span'].split(';'):
                    start, stop = s.split(' ')
                    entity += text[int(start):int(stop)]
            else:
                if entity in row:
                    if not pd.isnull(row['entity']):
                        entity = row['entity']

            # brat does not like spaces in the annotations
            # therefore we add a step to strip them from the string
            L = len(entity) - len(entity.lstrip(' '))
            R = len(entity) - len(entity.rstrip(' '))

            start = row['start'] + L
            stop = row['stop'] - R
            txt = entity[L:len(entity)-R]

            # brat does not like spaces in the type names
            typ = row['entity_type'].replace(' ', '_')

            # add in partial/missed annotations
            if row['partial'] == 1:
                partial_matches.append([row['span'], start, stop, txt])

            if row['missed'] == 1:
                missed.append([start, stop, txt])

            # no newlines in annotation! must literally write out '\n' or '\r'
            # N.B. brat 1.3 complains about text spanning multiple rows
            # if spacing is used, see issue #1040
            txt = txt.replace('\n', '\\n')
            txt = txt.replace('\r', '\\r')

            # all annotations are text-bound annotations, 'T'
            # therefore, each line formatted as:
            #   T(ann #), tab, type/start/end, tab, string for reference
            # middle is a space-separated triple
            #   (type, start-offset, end-offset)
            annotation = '{}\t{} {} {}\t{}\n'.format(
                a, typ, start, stop, txt)
            fp.write(annotation)

        t = max([int(x[1:]) for x in df['annotation_id']]) + 1

        # add in partial/missed annotations
        for span, start, stop, txt in partial_matches:
            # brat will automatically insert reference text
            # so we insert blank text ('') after the tab
            if ';' in span:
                segments = span.split(';')
            else:
                segments = [span]

            for segment in segments:
                ignore_partial, entity = suppress_partials(
                    txt, segment, start, stop)
                if ignore_partial:
                    continue

                entity = entity.replace('\n', '\\n')
                entity = entity.replace('\r', '\\r')
                annotation = 'T{}\t{} {}\t{}\n'.format(
                    t, 'p_missed', segment, entity)
                fp.write(annotation)
                t += 1

        for start, stop, txt in missed:
            txt = txt.replace('\n', '\\n')
            txt = txt.replace('\r', '\\r')
            annotation = 'T{}\t{} {} {}\t{}\n'.format(
                t, 'missed', start, stop, txt)
            fp.write(annotation)
            t += 1


def pattern_spans(text, pattern):
    """
    Iterator that splits text using a regex.
    Also returns the span of words (start and stop indices).
    """

    tokens = pattern.split(text)

    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)


def split_by_pattern(text, pattern):
    """
    Iterator that wraps the pattern span splitter.
    """
    tokens_with_spans = list()
    n = 0
    for token, start, end in pattern_spans(text, pattern):
        tokens_with_spans.append([n, start, end, token])
        n += 1

    return tokens_with_spans


def get_entity_token_context(df, text,
                             context=3):
    """
    Given an annotation dataframe, get nearby tokens

    context: how many words on each side to include (default 3).
    """
    if df.shape[0] == 0:
        return
    pattern = re.compile(r'\s')

    # create list of list, each sublist is:
    #   counter, start index, stop index, token
    text_split = split_by_pattern(text, pattern)

    # ensure df is in ascending order of indices
    df = df.sort_values(['start', 'stop'])
    annotations = list()
    for idx, row in df.iterrows():
        a = row['annotation_id']

        # we prefer to rederive entity from the text
        spans = row['span'].split(';')
        earliest_span_start = int(spans[0].split(' ')[0])
        last_span_stop = int(spans[-1].split(' ')[1])

        # find the first span which is after this entity
        # x[0] is the index of the row
        span_left_idx = next(
            (x[0] for x in text_split
                if x[2] >= earliest_span_start), None)

        if span_left_idx is None:
            # there is no entity after this span
            # span must be the last word
            # left/right span idx are set to the last token
            span_left_idx = text_split[-1][0]
            span_right_idx = text_split[-1][0]
        else:
            span_right_idx = next(
                (x[0] for x in text_split
                    if x[2] > last_span_stop), None)
            # span ends at the last token
            if span_right_idx is None:
                span_right_idx = text_split[-1][0]

        context_right = ['']*context
        # iterate through tokens on the right side of the span
        i = 0
        for s in range(span_right_idx, span_right_idx+context):
            # ensure we do not go beyond the text
            if s < text_split[-1][0]:
                # add word to context vector
                context_right[i] = text_split[s][3]
            i += 1

        context_left = ['']*context
        # iterate through tokens on the left side of the span
        i = -1
        for s in range(span_left_idx - 1, span_left_idx - context - 1, -1):
            # ensure we do not go beyond the text
            if s > 0:
                # add word to context vector
                context_left[i] = text_split[s][3]
            i -= 1

        annotation = [a, ';' in row['span']] + \
            context_left + [row['entity']] + context_right
        annotations.append(annotation)

    return annotations


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_entity_context(df, text, context=30, color=False):
    """
    Given an annotation dataframe, get nearby characters

    context: how many words on each side to include (default 3).
    """
    if df.shape[0] == 0:
        return

    text = text.replace('\n', ' ')

    # ensure df is in ascending order of indices
    df = df.sort_values(['start', 'stop'])
    annotations = list()
    for _, row in df.iterrows():
        # we prefer to rederive entity from the text
        spans = row['span'].split(';')
        for i in range(len(spans)):
            spans[i] = spans[i].split(' ')
            spans[i][0], spans[i][1] = int(spans[i][0]), int(spans[i][1])

        earliest_span_start = spans[0][0]
        last_span_stop = spans[-1][1]

        span_start = max(earliest_span_start - context, 0)
        span_stop = min(last_span_stop + context, len(text))

        # initialize annotation list with annotation_id
        annotation = [row['annotation_id']]

        # start with context before the first span
        annotation.append(text[span_start:earliest_span_start])
        i = 0
        while i < (len(spans)-1):
            # handle segments of entities
            s = spans[i]
            text_add = text[s[0]:s[1]]
            # bcolors.BOLD
            if color:
                text_add = bcolors.FAIL + text_add + bcolors.ENDC
            annotation.append(text_add)

            # also add in text between two entities
            next_span_start = spans[i+1][0]
            annotation.append(text[s[1]:next_span_start])
            i += 1

        # final entity span
        s = spans[-1]
        text_add = text[s[0]:s[1]]
        # bcolors.BOLD
        if color:
            text_add = bcolors.FAIL + text_add + bcolors.ENDC
        annotation.append(text_add)
        annotation.append(text[s[1]:span_stop])

        annotations.append(annotation)

    return annotations


def add_brat_conf_files(out_path):
    """
    Check for annotations.conf, visual.conf, and tools.conf
    in the brat folder. Add them if necessary.
    """
    # check for annotations.conf, add it if it doesn't exist
    fn = os.path.join(out_path, 'annotation.conf')
    if not os.path.exists(fn):
        annotations_conf = ['[entities]',
                            'Age',
                            'Name',
                            'Date',
                            'Identifier',
                            'Location',
                            'Organization',
                            'Nationality',
                            'Telephone',
                            'Address',
                            'Protected_Entity',
                            'Contact',
                            'p_missed',
                            'missed',
                            '',
                            '[relations]',
                            '[events]',
                            '[attributes]'
                            ]
        with open(fn, 'w') as fp:
            fp.write('\n'.join(annotations_conf))

    # specify how to display the annotations
    fn = os.path.join(out_path, 'visual.conf')
    if not os.path.exists(fn):
        visual_conf = ['[labels]', '[drawing]',
                       'SPAN_DEFAULT    fgColor:black, bgColor:lightgreen, borderColor:darken',
                       'ARC_DEFAULT     color:black, dashArray:-, arrowHead:triangle-5',
                       'p_missed bgColor:#ffff33',
                       'missed bgColor:#e41a1c',
                       ]
        with open(fn, 'w') as fp:
            fp.write('\n'.join(visual_conf))

    fn = os.path.join(out_path, 'tools.conf')
    if not os.path.exists(fn):
        tools_conf = ['[options]', 'Sentences	splitter:newline']
        with open(fn, 'w') as fp:
            fp.write('\n'.join(tools_conf))
