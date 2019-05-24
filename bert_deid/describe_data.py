# create a single CSV for training using data stored as gs/txt files
"""
Usage example (from root directory):
  python utils/summarize_dataset.py -i dernoncourt_goldstandard
"""
import argparse
import os
import sys
import re

import pandas as pd
from tqdm import tqdm

# our dataframe will have consistent columns
COLUMN_NAMES = ['document_id', 'annotation_id', 'start',
                'stop', 'entity', 'entity_type', 'comment']


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


def parse_arguments(args):
    ''' Parse the command line arguments.

    Args:
        arguments (command line arguments): Optional arguments specified at
            the command line.
    '''
    parser = argparse.ArgumentParser(description='Summarize a dataset')
    parser.add_argument('-i', '--input', type=str,
                        default=None, required=True,
                        help='Folder with ann and txt subfolders')
    parser.add_argument('-harmonize', '--harmonize', type=str,
                        default=True, required=False,
                        help='Harmonize annotation labels to i2b2 2014 categories')

    arguments = parser.parse_args(args)

    return arguments


def check_folder_exists(path):
    """
    Check whether a folder exists.
    """
    if not os.path.exists(path):
        raise ValueError(f'Could not find folder {path}')
    else:
        return 0


def get_record_names(txt_path, ann_path):
    """
    Gets a list of the record names.
    """
    # get a list of the directory contents
    ann_files = os.listdir(ann_path)
    txt_files = os.listdir(txt_path)

    # filter to files with correct extension // that overlap
    ann_files = [f for f in ann_files if f.endswith('.gs')]
    txt_files = set([f for f in txt_files if f.endswith('.txt')])
    records = [f[0:-3] for f in ann_files if f[0:-3] + '.txt' in txt_files]

    if len(records) == 0:
        sys.exit(f'No matched text files and annotations found.')
    else:
        return records


def create_report(annotations, notes, harmonize=True):
    """
    Creates the report.

    Args:
        annotations (dataframe): Dataframe of annotations in Pydeid format.
        notes (list): List of reports associated with the annotations.
        harmonize (bool): True to group entities into one of the i2b2 2014
            categories
    """
    header = "Summary"

    if harmonize:
        annotations['entity_type'] = annotations['entity_type'].map(
            harmonize_label)
        header = header + " (harmonized labels)"
    else:
        pass

    n_notes = len(notes)
    n_char = sum([len(x) for x in notes])
    n_ann = annotations.shape[0]

    tokenizer = re.compile(r'\s')
    tokens = [len(tokenizer.split(x)) for x in annotations['entity']]

    n_phi_tokens = sum(tokens)
    n_ann_gt_1_token = sum([x > 1 for x in tokens])
    prop_ann_gt_1_token = n_ann_gt_1_token / annotations.shape[0]*100.0

    print('\n{}\n{}\n{}'.format('-'*14, header, '-'*14))
    print(f'Records: {n_notes}.')
    print(f'Characters: {n_char}.')
    print(f'PHI annotations: {n_ann}.')
    print(f'PHI tokens: {n_phi_tokens}.')
    print(
        f'Annotations with >1 PHI token: {n_ann_gt_1_token} ({prop_ann_gt_1_token:4.1f}%).\n')
    print(annotations['entity_type'].value_counts())
    print('{}\n'.format('-'*14))


def collect_records(input_path):
    """
    Collects the annotations and notes into two separate files.

    Args:
        input_path (str): path to the folder containing the text and annotations.
            The folder must contain folders named 'ann' and 'txt'.

    Returns:
        annotations (dataframe): a dataframe containing the annotations.
        notes (list): a list of text files associated with the annotations.
    """
    ann_path = os.path.join(input_path, 'ann')
    txt_path = os.path.join(input_path, 'txt')
    check_folder_exists(input_path)
    check_folder_exists(ann_path)
    check_folder_exists(txt_path)
    records = get_record_names(txt_path, ann_path)

    notes = list()
    ann_df = list()

    print('Loading dataset...')
    for doc_id in tqdm(records):
        # read PHI annotations
        df = pd.read_csv(os.path.join(ann_path, doc_id + '.gs'),
                         dtype={'entity': str})
        ann_df.append(df)

        with open(os.path.join(txt_path, doc_id + '.txt'), 'r') as fp:
            text = ''.join(fp.readlines())
        notes.append(text)

    annotations = pd.concat(ann_df, ignore_index=True)

    return annotations, notes


def pretty_print_report(text, text_start, start, stop, color_text):
    # print colored text

    # reset the color
    print('\033[0m', end='')

    print(text[text_start:start], end='')
    # print text between start/stop in the format of our choice
    print(color_text, end='')
    print(text[start:stop], end='')

    # reset the color
    print('\033[0m', end='')

    # update start so the next iteration prints it in normal color
    return stop


def pretty_print_comparison(text, deid, gs):
    """
    Prints a report, with identified PHI in green and missed PHI in bolded red.
    """

    # green
    color_good = '\033[92m'

    # bolded red
    color_bad = '\033[91m\033[1m'

    deid.sort_values(['start', 'stop'], inplace=True)
    gs.sort_values(['start', 'stop'], inplace=True)

    n_deid = deid.shape[0]
    n_gs = gs.shape[0]

    text_start = 0
    i = 0  # deid iterator
    g = 0  # gold standard iterator
    d_start, d_stop = deid.iloc[i, :].values
    g_start, g_stop = gs.iloc[g, :].values

    while (g < n_gs) | (i < n_deid):

        # where does the first instance of phi/deid start/stop
        start = min(d_start, g_start)
        stop = min(d_stop, g_stop)

        # decide what color - if deid before gold standard, green
        if d_start <= g_start:
            print_color = color_good
        else:
            # if gold standard before deid, we missed it -> red
            print_color = color_bad

        # print text up to 'start' as normal
        # print text between start/stop in the color of our choice
        text_start = pretty_print_report(
            text, text_start, start, stop, print_color
        )

        # update deid/gs offsets if we have passed them
        if text_start >= g_stop:
            g += 1
            if g < n_gs:
                g_start, g_stop = gs.iloc[g, :].values
            else:
                g_start, g_stop = len(text), len(text)

        if text_start >= d_stop:
            i += 1
            if i < n_deid:
                d_start, d_stop = deid.iloc[i, :].values
            else:
                d_start, d_stop = len(text), len(text)

    if len(text) > text_start:
        print(text[text_start:])


def main(args):
    """
    Command line interface for summarizing a dataset.
    """
    arguments = parse_arguments(args)

    # iterate files in the input folder.
    # add annotations to a dataframe and add notes to a list
    annotations, notes = collect_records(input_path=arguments.input)
    harmonize = True if arguments.harmonize.lower() in [
        "true", "yes"] else False
    create_report(annotations, notes, harmonize)


if __name__ == '__main__':
    main(sys.argv[1:])
