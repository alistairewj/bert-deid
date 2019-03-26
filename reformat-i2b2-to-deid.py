# reformats i2b2 de-identification task data into our format


import argparse
import os
import sys
import re
import csv
import itertools
import xml.etree.ElementTree as ET

# must have installed punkt model
# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert i2b2 annotations')
parser.add_argument('-p', '--path', dest='path', type=str,
                    default='train',
                    help='location of XML files to convert')
parser.add_argument('-o', '--output', type=str,
                    default='output',
                    help='folder to output converted annotations')

# create a custom iterator that also returns the span of the sentence


def sentence_spans(text):
    tokens = sent_tokenize(text)

    # further split using token '\n \n '
    tokens = [x.split('\n \n ') for x in tokens]
    # flatten sublists into a single list
    tokens = list(itertools.chain.from_iterable(tokens))

    # further split using token '\n\n'
    tokens = [x.split('\n\n') for x in tokens]
    tokens = list(itertools.chain.from_iterable(tokens))

    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)


def tokenize_sentence(text):
    # section the reports
    # p_section = re.compile('^\s+([A-Z /,-]+):', re.MULTILINE)
    sentences = list()
    n = 0
    for sent, start, end in sentence_spans(text):
        sentences.append([n, start, end, sent])
        n += 1

    return sentences


def main(args):
    args = parser.parse_args(args)

    # for each file in the folder
    base_path = args.path

    out_path = args.output

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(os.path.join(out_path, 'ann')):
        os.mkdir(os.path.join(out_path, 'ann'))
    if not os.path.exists(os.path.join(out_path, 'txt')):
        os.mkdir(os.path.join(out_path, 'txt'))

    files = os.listdir(base_path)
    files = [f for f in files if f.endswith('.xml')]

    if len(files) == 0:
        print(f'No files found in folder {base_path}')
        return

    tag_list = ['id', 'start', 'end', 'text', 'TYPE', 'comment']
    # we rename the columns to be consistent with other deid dataset
    #
    col_names = ['document_id', 'annotation_id', 'start',
                 'stop', 'entity', 'entity_type', 'comment']

    deid_all = list()
    sentences_all = list()
    deid_types = list()

    for f in files:
        # load as XML tree
        fn = os.path.join(base_path, f)
        with open(fn, 'r', encoding='UTF-8') as fp:
            xml_data = fp.read()
        tree = ET.fromstring(xml_data)

        # get the text from TEXT field
        text = tree.find('TEXT')
        if text is not None:
            text = text.text
        else:
            print(f'WARNING: {fn} did not have any text.')

        # the <TAGS> section has deid annotations
        tags_xml = tree.find('TAGS')

        # example tag:
        # <DATE id="P0" start="16" end="26" text="2069-04-07" TYPE="DATE" comment="" />
        tags = list()
        for tag in tags_xml:
            tags.append([f] + [tag.get(t) for t in tag_list])

        # create dataframe of all PHI tags
        df = pd.DataFrame(tags, columns=col_names)
        df['start'] = df['start'].astype(int)
        df['stop'] = df['stop'].astype(int)

        # output dataframe style PHI
        document_id = f[0:-4]
        df.to_csv(os.path.join(out_path, 'ann',
                               document_id + '.gs'), index=False)
        with open(os.path.join(out_path, 'txt', document_id + '.txt'), 'w') as fp:
            fp.write(text)

        # split the text into sentences
        sentences = tokenize_sentence(text)

        # match annotations to the appropriate sentence
        sIdx = 0
        deid_info = [list() for x in range(len(sentences))]
        for i, row in df.iterrows():
            # go to the sentence which ends after this annotation begins
            # [s][2] is the end offset of the sentence
            deid_type = row['entity_type']

            # HACK: ensure deid_types are members of a fixed set
            """
            if deid_type in ('Age > 89', 'Age_>_89'):
                deid_type = 'Age'
            elif deid_type == 'Nationality':
                deid_type = 'Location'
            elif deid_type in ('Protected Entity', 'Organization'):
                deid_type = 'Protected_Entity'
            """
            deid_types.append(deid_type)

            while sentences[sIdx][2] < row['start']:
                sIdx += 1

            s_start, s_stop = sentences[sIdx][1], sentences[sIdx][2]
            if row['stop'] <= s_stop:
                # add indices of annotated de-id
                # adjust them based upon sentence start offset
                deid_info[sIdx].append(
                    [row['start'] - s_start, row['stop'] - s_start, deid_type])
            else:
                # if we are here, the PHI row starts in this sentence..
                # but ends in the next sentence. this is an edge case.
                # split the row annotation across the two sentences
                deid_info[sIdx].append(
                    [row['start'] - s_start, s_stop - s_start, deid_type])

                sIdx += 1
                deid_info[sIdx].append(
                    [0, row['stop'] - sentences[sIdx][1], deid_type])

        # save to master list
        sentences_all.extend([[document_id] + x for x in sentences])
        deid_all.extend(deid_info)

    # create sentence-wise deid format with all sentence data
    deid_fn = os.path.join(out_path, 'sentences.csv')
    with open(deid_fn, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(deid_all)):
            sentence = sentences_all[i]
            # create row: text id, sentence, list of PHI indices
            row = ['.'.join([str(x) for x in sentence[:4]]),
                   sentence[4], deid_all[i]]
            csvwriter.writerow(row)


if __name__ == '__main__':
    main(sys.argv[1:])
