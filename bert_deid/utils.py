import os
import warnings
import re

from sympy import Interval, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
import stanfordnlp
import spacy
from nltk import word_tokenize


def combine_entity_types(df, lowercase=True):
    """
    Merge together similar entity types.

    Inputs:
    df (dataframe): annotation dataframe
    lowercase (boolean): output all entities as lowercase.
    """

    # coalesce pydeid types to types understood by the annotation tool
    # mainly this converts all name related labels to "Name"
    type_dict = {'initials': 'Name', 'firstname': 'Name', 'lastname': 'Name'}
    df['entity_type'] = df['entity_type'].apply(
        lambda x: type_dict[x] if x in type_dict else x
    )

    # use the annotator name to infer the entity type
    type_fix = {'wmrn': 'Identifier', 'name': 'Name', 'initials': 'Name'}
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
        interval = Interval(interval.left + 1, interval.right)
    if not interval.right_open:
        interval = Interval(interval.left, interval.right + 1, False, True)
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

    df.sort_values(
        ['document_id', 'entity_type', 'start', 'stop'], inplace=True
    )
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
                    entity += '?' * -update_stop + row['entity']
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
            warnings.warn(
                'Added ? characters to entities for unknown characters',
                category=RuntimeWarning
            )
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
    df = pd.concat(
        [goldstandard[cols], comparison[cols]], ignore_index=True, axis=0
    )

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
        cmp_intervals = [
            [x[0], x[1], False, True]
            for x in grp.loc[~idxG, ['start', 'stop']].values
        ]
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
                    [grp_idx, row['annotation_id'], 1, 0, 0, span]
                )
            # partial match
            else:
                # no match
                if mismatch == i:
                    span = '{} {}'.format(row['start'], row['stop'])
                    performance.append(
                        [grp_idx, row['annotation_id'], 0, 0, 1, span]
                    )
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
                        [grp_idx, row['annotation_id'], 0, 1, 0, span]
                    )

    # convert back to a dataframe with same index as gs
    performance = pd.DataFrame.from_records(
        performance,
        columns=[
            'document_id', 'annotation_id', 'exact', 'partial', 'missed', 'span'
        ]
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
                    [doc_id, row['annotation_id'], 0, 0, 1, span]
                )

        # if ann.shape[0] == 0:
        #     # append ann rows as false positives
        #     for i, row in ann.iterrows():
        #         span = '{} {}'.format(row['start'], row['stop'])
        #         performance.append(
        #             [doc_id, row['annotation_id'], 0, 0, 0, span])

        performance = pd.DataFrame.from_records(
            performance,
            columns=[
                'document_id', 'annotation_id', 'exact', 'partial', 'missed',
                'span'
            ]
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
    cmp_intervals = [
        [x[0], x[1], False, True] for x in ann.loc[:, ['start', 'stop']].values
    ]
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
            performance.append([doc_id, row['annotation_id'], 1, 0, 0, span])
        # partial match
        else:
            # no match
            if mismatch == i:
                span = '{} {}'.format(row['start'], row['stop'])
                performance.append(
                    [doc_id, row['annotation_id'], 0, 0, 1, span]
                )
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
                    [doc_id, row['annotation_id'], 0, 1, 0, span]
                )

    # convert back to a dataframe with same index as gs
    performance = pd.DataFrame.from_records(
        performance,
        columns=[
            'document_id', 'annotation_id', 'exact', 'partial', 'missed', 'span'
        ]
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
            idx = [
                idx[i] != df['annotation_id'].values[i]
                for i in range(df.shape[0])
            ]
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
            txt = entity[L:len(entity) - R]

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
            annotation = '{}\t{} {} {}\t{}\n'.format(a, typ, start, stop, txt)
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
                    txt, segment, start, stop
                )
                if ignore_partial:
                    continue

                entity = entity.replace('\n', '\\n')
                entity = entity.replace('\r', '\\r')
                annotation = 'T{}\t{} {}\t{}\n'.format(
                    t, 'p_missed', segment, entity
                )
                fp.write(annotation)
                t += 1

        for start, stop, txt in missed:
            txt = txt.replace('\n', '\\n')
            txt = txt.replace('\r', '\\r')
            annotation = 'T{}\t{} {} {}\t{}\n'.format(
                t, 'missed', start, stop, txt
            )
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
        yield token, offset, offset + len(token)
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


def get_entity_token_context(df, text, context=3):
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
            (x[0] for x in text_split if x[2] >= earliest_span_start), None
        )

        if span_left_idx is None:
            # there is no entity after this span
            # span must be the last word
            # left/right span idx are set to the last token
            span_left_idx = text_split[-1][0]
            span_right_idx = text_split[-1][0]
        else:
            span_right_idx = next(
                (x[0] for x in text_split if x[2] > last_span_stop), None
            )
            # span ends at the last token
            if span_right_idx is None:
                span_right_idx = text_split[-1][0]

        context_right = [''] * context
        # iterate through tokens on the right side of the span
        i = 0
        for s in range(span_right_idx, span_right_idx + context):
            # ensure we do not go beyond the text
            if s < text_split[-1][0]:
                # add word to context vector
                context_right[i] = text_split[s][3]
            i += 1

        context_left = [''] * context
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
        while i < (len(spans) - 1):
            # handle segments of entities
            s = spans[i]
            text_add = text[s[0]:s[1]]
            # bcolors.BOLD
            if color:
                text_add = bcolors.FAIL + text_add + bcolors.ENDC
            annotation.append(text_add)

            # also add in text between two entities
            next_span_start = spans[i + 1][0]
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
        annotations_conf = [
            '[entities]', 'Age', 'Name', 'Date', 'Identifier', 'Location',
            'Organization', 'Nationality', 'Telephone', 'Address',
            'Protected_Entity', 'Contact', 'p_missed', 'missed', '',
            '[relations]', '[events]', '[attributes]'
        ]
        with open(fn, 'w') as fp:
            fp.write('\n'.join(annotations_conf))

    # specify how to display the annotations
    fn = os.path.join(out_path, 'visual.conf')
    if not os.path.exists(fn):
        visual_conf = [
            '[labels]',
            '[drawing]',
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


def split_by_token_entity(text, entities, start):
    """
    Split a token with conflict entity type
        i.e. a token "Home/Bangdung" with "Home" as HOSPITAL, "/" as object, "Bangdung" as CITY
        would be splitted into three tokens: "Home", "/", "Bandung"
    """
    prev_type = entities[0]
    tokens, starts, ends = [], [], []
    offset = 0
    for i in range(len(text)):
        if entities[i] != prev_type:
            token = text[offset:i]
            tokens.append(token)
            starts.append(offset + start)
            ends.append(offset + len(token) + start)

            offset += len(token)
            prev_type = entities[i]
    last_token = text[offset:len(text)]
    tokens.append(last_token)
    starts.append(offset + start)
    ends.append(offset + len(last_token) + start)
    return tokens, starts, ends


def split_by_space(text):
    """
    Split a corpus by whitespace/new line to get token, corresponding start index and end index
    """
    offset = 0
    for token in text.split():
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)

def split_by_space_punctuation(text):
    offset = 0
    for token in re.findall(r"[\w]+|[^\s\w]", text):
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)


def split_by_nltk(text):
    offset = 0
    for token in word_tokenize(text):
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)


def compute_stats(df, token_eval=True, average='micro', label=None):
    '''
    Compute se (recall), ppv (precision), f1 score
    Args:
        df: DataFrame, contains stats for each file in dataset
        token_eval: Bool, token or entity evaultion
        average: string, ['micro', 'macro']
        label: string, None (if binary evalution), valid label (if multi evaluation)
    '''
    average = average.lower()
    if average not in ('micro', 'macro'):
        raise ValueError('Invalid average argument.')
    if label is not None:
        type_eval = f'n_{label}_'
    else:
        type_eval = 'n_'

    if token_eval:
        type_eval += 'token_'
    else:
        type_eval += 'entity_'

    tp = df[type_eval + 'tp']
    fp = df[type_eval + 'fp']
    fn = df[type_eval + 'fn']
    if average == 'micro':
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    se = tp / (tp + fn)
    ppv = tp / (tp + fp)
    f1 = 2 * se * ppv / (se + ppv)
    return se, ppv, f1


def get_entities(data):
    """
    Get PHI entities (entity, entity type, start, stop) from dataframe
    """
    entities = [
        (
            data['entity'].iloc[i], data['entity_type'].iloc[i],
            data['start'].iloc[i], data['stop'].iloc[i]
        ) for i in range(len(data))
    ]

    return entities


def ignore_partials(phis):
    """
    Create a new data phis that ignore punctuation at front/end of entity
    """
    # phi set: (entity, entity_type, start, stop)
    partials = (' ', '\r', '\n', ';', ':', '-')
    new_phis = []
    for (entity, entity_type, start, stop) in phis:
        if entity[0] in partials:
            entity = entity[1:]
            start += 1

        if len(entity) > 0:
            if entity[-1] in partials:
                entity = entity[:-1]
                stop += 1
        new_phis.append((entity, entity_type, start, stop))
    return new_phis


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


def tokenize_text(tokenizer, text):
    """Split text into tokens using the given tokenizer."""
    if isinstance(tokenizer, stanfordnlp.pipeline.core.Pipeline):
        doc = tokenizer(text)
        # extract tokens from the parsed text
        tokens = [
            token.text for sentence in doc.sentences
            for token in sentence.tokens
        ]
    elif isinstance(tokenizer, spacy.tokenizer.Tokenizer):
        doc = tokenizer(text)
        # extract tokens from the parsed text
        tokens = [token.text for token in doc]
    else:
        if tokenizer is None:
            tokenizer = r'\w'
        # treat string as a regex
        tokens = re.findall(tokenizer, text)
    return tokens


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
    tokens_base = tokenize_text(tokenizer, text)

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
