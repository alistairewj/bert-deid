# reformat deid gs sub-folder into BioC format
# example usage:
#   python scripts/deid_gs_to_bioc.py -p /db/git/deid-gs/dernoncourt_goldstandard/train -o /home/alistairewj/git/annotation-tool/data/dl_train
import argparse
import os
import sys
import re
import csv
import itertools
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert deid-gs folder to xml')
parser.add_argument('-p', '--path', dest='path', type=str,
                    default='train',
                    help='location of XML files to convert')
parser.add_argument('-o', '--output', type=str,
                    default='output',
                    help='folder to output converted annotations')
parser.add_argument('-s', '--standardize',
                    action='store_true',
                    help='rewrite entity types into standard categories')


def df2bioc(df, text=None, output=None):
    """Given a text and a dataframe of annotations, output an XML file.

    The XML format is BioC.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3889917/
    https://github.com/2mh/PyBioC/blob/master/BioC.dtd

    Here is an example of an XML file in this format:

        <?xml version='1.0' encoding='UTF-8'?>
        <!DOCTYPE bioc SYSTEM 'https://github.com/2mh/PyBioC/blob/master/BioC.dtd'>
        <collection>
            <source></source>
            <date></date>
            <key></key>
            <document>
                <id>1000</id>
                <passage>
                    <offset>0</offset>
                    <text> ...
                    </text>
                    <annotation id='0'>
                        <infon key='type'>Protected Entity</infon>
                        <location length='36' offset='18'/>
                        <text>BETH ISRAEL DEACONESS MEDICAL CENTER</text>
                    </annotation>
                </passage>
            </document>
        </collection>

    Returns a string containing the XML.
    """
    cols = ['document_id', 'annotator', 'start',
            'stop', 'entity', 'entity_type']

    for i, c in enumerate(cols):
        if c not in df.columns:
            raise Exception('Missing column "{}" in dataframe.'.format(c))

    collection = ET.Element('collection')
    # source = ET.SubElement(collection, 'source')
    # date = ET.SubElement(collection, 'date')
    # key = ET.SubElement(collection, 'key')

    # add each document_id
    for d in df['document_id'].unique():
        # init the document subelement
        document = ET.SubElement(collection, 'document')

        # add ID
        doc_id = ET.SubElement(document, 'id')
        doc_id.text = str(d)

        # add passage - only one passage per doc output by this function
        passage = ET.SubElement(document, 'passage')
        offset = ET.SubElement(passage, 'offset')
        offset.text = '0'

        # add text if input by user
        text_node = ET.SubElement(passage, 'text')
        if text is not None:
            text_node.text = text

        for i, row in df.loc[df['document_id'] == d, :].iterrows():
            annotation = ET.SubElement(passage, 'annotation')
            annotation.attrib = {'id': str(i)}

            for c in ('annotator', 'entity_type'):
                if pd.isnull(row[c]):
                    continue
                infon = ET.SubElement(annotation, 'infon')
                if c == 'entity_type':
                    # HACK: hard code entity type as type
                    infon.attrib = {'key': 'type'}
                else:
                    infon.attrib = {'key': c}
                infon.text = str(row[c])

            # add location/text of annotation
            location = ET.SubElement(annotation, 'location')
            location.attrib = {'length': str(row['stop']-row['start']),
                               'offset': str(row['start'])}

            text_node = ET.SubElement(annotation, 'text')
            if not pd.isnull(row['entity']):
                text_node.text = row['entity']
            else:
                text_node.text = ''

    # initialize with encoding info
    xml_str = "<?xml version='1.0' encoding='UTF-8'?>"
    xml_str += "\n"
    url = 'https://github.com/2mh/PyBioC/blob/master/BioC.dtd'
    xml_str += "<!DOCTYPE bioc SYSTEM '{}'>".format(url)
    xml_str += "\n"

    # add the nodes - manually decode to avoid tostring() writing encoding info
    xml_str += ET.tostring(collection, method='xml').decode('utf-8')

    if output is not None:
        # output to this file
        with open(output, 'w') as fp:
            fp.write(xml_str)

    return xml_str


def main(args):
    args = parser.parse_args(args)

    # for each file in the folder
    base_path = args.path
    txt_path = os.path.join(base_path, 'txt')
    ann_path = os.path.join(base_path, 'ann')
    out_path = args.output

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    files = os.listdir(txt_path)
    files = [f[0:-4] for f in files if f.endswith('.txt')]
    files = set(files)

    ann_files = os.listdir(ann_path)
    ann_files = [f[0:-3] for f in ann_files if f.endswith('.gs')]
    ann_files = set(ann_files)

    files = files.intersection(ann_files)
    files = list(files)

    if len(files) == 0:
        print(f'No files found.')
        return

    # default harmonization is into i2b2 form
    if args.standardize:
        labels = [
            ['Name', ['NAME', 'DOCTOR', 'PATIENT', 'USERNAME', 'HCPNAME',
                      'RELATIVEPROXYNAME', 'PTNAME', 'PTNAMEINITIAL']],
            ['Protected Entity', ['LOCATION', 'HOSPITAL', 'ORGANIZATION',
                                  'STREET', 'STATE',
                                  'CITY', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
                                  'PROTECTED_ENTITY', 'PROTECTED ENTITY',
                                  'NATIONALITY']],
            ['Age > 89', ['AGE', 'AGE_>_89', 'AGE > 89']],
            ['Date', ['DATE', 'DATEYEAR']],
            ['Identifier', ['BIOID', 'DEVICE', 'HEALTHPLAN',
                            'IDNUM', 'MEDICALRECORD', 'ID', 'OTHER']],
            ['Contact', ['EMAIL', 'FAX', 'PHONE', 'URL',
                         'IPADDR', 'IPADDRESS', 'CONTACT']],
            ['Other', ['PROFESSION']]
        ]

        label_to_type = {}
        for label_map in labels:
            for l in label_map[1]:
                label_to_type[l] = label_map[0]

    # loop through phi files
    print('Converting to XML...')
    for fn in tqdm(files, total=len(files)):
        # read text
        with open(os.path.join(txt_path, f'{fn}.txt'), 'r') as fp:
            text = ''.join(fp.readlines())

        df = pd.read_csv(os.path.join(ann_path, f'{fn}.gs'),
                         sep=',', header=0)

        df['annotator'] = 'gs'

        # map entity types to work with annotation tool
        if args.standardize:
            df['entity_type'] = df['entity_type'].map(label_to_type)

        # convert to the XML and write out
        df2bioc(
            df,
            text=text,
            output=os.path.join(out_path, f'{fn}.xml')
        )


if __name__ == '__main__':
    main(sys.argv[1:])
