"""
creates a "training_set_tokens.csv" file
sourced from the training dataset.
"""

import json
import sys
import os
import re
import argparse

import pandas as pd


def main(args):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('-c', "--config",
                        required=True,
                        type=str,
                        help=("Configuration file (json)."
                              "Specifies folders and parameters."))

    args = parser.parse_args(args)

    # if config provided, update dict with values from config
    with open(args.config, 'r') as fp:
        config = json.load(fp)

    if 'input_path' not in config:
        raise ValueError('Input path must be in config file.')
    input_path = config['input_path']

    if 'text_extension' in config:
        txt_ext = config['text_extension']
        if txt_ext[0] != '.':
            txt_ext = '.' + txt_ext
    else:
        txt_ext = '.txt'

    if 'ref_extension' in config:
        ann_ext = config['ref_extension']
        if ann_ext[0] != '.':
            ann_ext = '.' + ann_ext
    else:
        ann_ext = '.gs'

    # check input folders exist
    ann_path = os.path.join(input_path, 'ann')
    txt_path = os.path.join(input_path, 'txt')
    if not os.path.exists(input_path):
        raise ValueError(f'Could not find folder {input_path}')
    if not os.path.exists(ann_path):
        raise ValueError(f'Could not find folder {ann_path}')
    if not os.path.exists(txt_path):
        raise ValueError(f'Could not find folder {txt_path}')

    ann_files = os.listdir(ann_path)
    txt_files = os.listdir(txt_path)

    # filter to files with correct extension // that overlap
    ann_files = set([f[0:-len(ann_ext)]
                     for f in ann_files
                     if f.endswith(ann_ext)])
    txt_files = set([f[0:-len(txt_ext)]
                     for f in txt_files
                     if f.endswith(txt_ext)])
    records = list(ann_files.intersection(txt_files))

    if len(records) == 0:
        print(f'No files with both text and annotations.')
        return

    all_tokens = {}
    pattern = re.compile(r'\s')
    stopchar_rem = re.compile(r'[,.;_\\\/?!@#$%^&*()-]')
    for doc_id in records:
        with open(os.path.join(txt_path, f'{doc_id}{txt_ext}'), 'r') as fp:
            text = ''.join(fp.readlines())

        tokens = pattern.split(text)
        for t in tokens:
            # remove boring characters
            t = stopchar_rem.sub('', t)
            if len(t) > 0:
                try:
                    all_tokens[t] += 1
                except KeyError:
                    all_tokens[t] = 1

    # write out to file using pandas for convenience
    df = pd.DataFrame.from_dict(all_tokens, orient='index')
    df.columns = ['count']
    df.to_csv(os.path.join(config['model_path'], 'training_set_tokens.csv'))


if __name__ == "__main__":
    main(sys.argv[1:])
