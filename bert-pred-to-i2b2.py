# convert predictions output by BERT code to i2b2 XML format


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

# use i2b2 processor to get label mapping of predictions
from bert_ner import i2b2Processor

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# custom tokenizer with subword tracking
from tokenization import BertTokenizerNER

parser = argparse.ArgumentParser(description='Convert BERT predictions to XML')
parser.add_argument('-p', '--path', type=str,
                    default='eval_predictions.csv',
                    help='location of BERT predictions')
parser.add_argument('-d', '--data', type=str,
                    default='i2b2',
                    help='location of original text files')
parser.add_argument('-o', '--output', type=str,
                    default='output',
                    help='folder to output converted annotations')

# parameters required for tokenizer
parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")


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

    # load predictions
    if not os.path.exists(args.path):
        raise FileNotFoundError(f'Could not find {args.path}')

    # load CSV file
    guids = []
    preds = []
    with open(args.path, 'r') as fp:
        csvreader = csv.reader(fp, delimiter=',')
        for row in csvreader:
            guids.append(row[0])
            preds.append(row[1:])

    # extract out filename from guid
    dataset, doc_id, s_num, start, stop = [], [], [], [], []
    for x in guids:
        x = x.split('-')
        dataset.append(x[0])
        # reconstitute the ID - handles filenames with hyphens
        row = '-'.join(x[1:])
        row = row.split('.')

        stop.append(row[-1])
        start.append(row[-2])
        s_num.append(row[-3])

        # reconstitute doc ID - handles filenames with dots
        doc_id.append('.'.join(row[0:-3]))

    # create a pandas dataframe
    df = pd.DataFrame(guids)
    df.columns = ['guid']

    df['dataset'] = dataset
    df['document_id'] = doc_id
    df['s_num'], df['start'], df['stop'] = s_num, start, stop
    df['s_num'] = df['s_num'].astype(int)
    df['start'] = df['start'].astype(int)
    df['stop'] = df['stop'].astype(int)

    # add predictions as series, with each element having the list
    df['pred'] = preds

    df.sort_values(['document_id', 's_num', 'start'],
                   ascending=True, inplace=True)

    # create tokenizer that was used for model training/evaluation
    tokenizer = BertTokenizerNER.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    # get map between predicted labels and human interpretable tags
    processor = i2b2Processor()
    label_list = processor.get_labels()
    # convert keys to string since preds are loaded in as string
    label_map = {str(i): label for i, label in enumerate(label_list)}

    # i2b2 XML has a hierarchical type for each label
    # the evaluation scripts expect XML tags to be one of 6 types
    label_to_type = {
            # names
            'DOCTOR': 'NAME',
            'PATIENT': 'NAME',
            'USERNAME': 'NAME',
            # professions
            'PROFESSION': 'PROFESSION',
            # locations
            'HOSPITAL': 'LOCATION',
            'ORGANIZATION': 'LOCATION',
            'STREET': 'LOCATION',
            'STATE': 'LOCATION',
            'CITY': 'LOCATION',
            'COUNTRY': 'LOCATION',
            'ZIP': 'LOCATION',
            'LOCATION-OTHER': 'LOCATION',
            # ages
            'AGE': 'AGE',
            # dates
            'DATE': 'DATE',
            # IDs
            'BIOID': 'ID',
            'DEVICE': 'ID',
            'HEALTHPLAN': 'ID',
            'IDNUM': 'ID',
            'MEDICALRECORD': 'ID',
            # CONTACT
            'EMAIL': 'CONTACT',
            'FAX': 'CONTACT',
            'PHONE': 'CONTACT',
            'URL': 'CONTACT',
            # note: IP address included here, but isn't in i2b2 data
            'IPADDRESS': 'CONTACT'
    }

    # some lists for debugging
    truncated_sentences = []
    truncated_doc = []

    for doc_id, grp in df.groupby('document_id'):
        # load original text from XML
        fn = os.path.join(args.data, doc_id + '.xml')
        if not os.path.exists(fn):
            print(f'{fn} - could not find file')
            continue

        # load as XML tree
        with open(fn, 'r', encoding='UTF-8') as fp:
            xml_data = fp.read()
        tree = ET.fromstring(xml_data)

        # get the text from TEXT field
        text = tree.find('TEXT')
        if text is not None:
            text = text.text
        else:
            print(f'{fn} - did not have any text.')
            continue

        # initialize the output XML
        xml_out = ET.Element('deIdi2b2')

        # add the text
        xml_text = ET.SubElement(xml_out, 'TEXT')
        xml_text.text = text

        # prepare subelement to include all tagged PHI
        xml_tags = ET.SubElement(xml_out, 'TAGS')

        # split the text into sentences
        # this is a list of lists, each sub-list has 4 elements:
        #   sentence number, start index, end index, text of the sentence
        sentences = tokenize_sentence(text)

        # convert this into a dictionary indexed by the sentence number
        sentences = {x[0]: x[1:] for x in sentences}

        # iterate through our dataframe
        for i, row in grp.iterrows():
            sent = sentences[row['s_num']]

            # tokenize text according to BERT
            tokens, tokens_sw, tokens_idx = tokenizer.tokenize(sent[2])

            # use indices to map predictions back
            # note we ignore the [CLS] tag at beginning of sentence
            pred = row['pred'][1:]

            if len(tokens) >= len(pred):
                # truncate tokens to max sequence length
                print(f'{doc_id} {i} - truncating from', end=' ')
                print('{} to {}'.format(len(tokens), len(pred) - 1))

                # drop last token as it is [SEP]
                pred = pred[0:-1]

                # debug output
                truncated_sentences.append(' '.join(tokens))
                truncated_doc.append(doc_id)

                tokens = tokens[0:len(pred)]
                tokens_sw = tokens_sw[0:len(pred)]
                tokens_idx = tokens_idx[0:len(pred)]

            else:
                pred = pred[0:len(tokens)]

            if len(pred) == 0:
                continue

            # the model does not make predictions for sub-words
            # the first word-part for a segmented sub-word is used as the prediction
            # so we append sub-word indices to previous non-subword indices
            j = 0
            while j < len(tokens_sw):
                if tokens_sw[j] == 1:
                    # pop these indices
                    idx = tokens_idx.pop(j)
                    # add sub-word index to previous non-subword
                    tokens_idx[j-1].extend(idx)
                    # remove the token from other lists
                    pred.pop(j)
                    tokens_sw.pop(j)
                    tokens.pop(j)
                else:
                    j += 1

            # tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            # tokens_sw = [0] + tokens_a_sw + [0]
            # indices = [[-1]] + tokens_a_idx + [[-1]]

            # use label_map to get free-text version of labels
            pred = [label_map[x] for x in pred]

            # loop through preds and create tags
            for p, pred_tag in enumerate(pred):
                # do not create tags for classifications of type 'O'
                if pred_tag == 'O':
                    continue

                # for this sentence, get start/stop indices
                # we will make these relative to the entire text by adding sent[0]
                start = tokens_idx[p][0]
                stop = tokens_idx[p][-1] + 1

                # add location/text of annotation
                tag = ET.SubElement(xml_tags, label_to_type[pred_tag])
                tag.attrib = {'id': f'P{p}',
                              'start': str(sent[0] + start),
                              'end': str(sent[0] + stop),
                              'text': sent[2][start:stop],
                              'TYPE': pred_tag,
                              'comment': ""
                              }

        # output XML as string
        xml_str = ET.tostring(xml_out, method='xml')
        output_fn = os.path.join(args.output, doc_id + '.xml')
        with open(output_fn, 'wb') as fp:
            fp.write(xml_str)

    # debug df
    se = pd.DataFrame(truncated_doc)
    se.columns = ['document_id']
    se['sentences'] = truncated_sentences
    se.to_csv('debug_truncated_sentences.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
