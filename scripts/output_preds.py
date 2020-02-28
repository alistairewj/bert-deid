import csv
import os
import pickle
from pathlib import Path
from bisect import bisect_left, bisect_right
import logging

import numpy as np
from transformers import BertTokenizer
from bert_deid.model import Transformer
from bert_deid.processors import load_labels, label_transform
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import argparse

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
        help="Type of dataset",
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
    parser.add_argument(
        "--label_transform",
        action="store_true",
        help="Whether labels are transformed using BIO"
    )
    args = parser.parse_args()

    # load in a trained model
    transformer = Transformer(
        args.model_type,
        args.model_dir,
        max_seq_length=128,
        task_name=args.task,
        cache_dir=None,
        device='cpu',
        label_transform=args.label_transform
    )
    label_to_id = {v: k for k, v in transformer.label_id_map.items()}

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

    files = os.listdir(data_path / 'txt')
    files = [f for f in files if f.endswith('.txt')]
    data = []

    preds = None
    lengths = []
    offsets = []
    labels = []
    # HACK: map physionet -> i2b2_2014
    pn_to_i2b2 = {
        'O': 'O',
        # names
        'HCPNAME': 'DOCTOR',
        'RELATIVEPROXYNAME': 'PATIENT',
        'PTNAME': 'PATIENT',
        'PTNAMEINITIAL': 'PATIENT',
        # professions
        # locations
        'LOCATION': 'LOCATION',
        # ages
        'AGE': 'AGE',
        # dates
        'DATE': 'DATE',
        'DATEYEAR': 'DATE',
        # IDs
        'OTHER': 'IDNUM',
        # contacts
        'PHONE': 'PHONE'
    }

    for f in tqdm(files, total=len(files), desc='Files'):
        with open(data_path / 'txt' / f, 'r') as fp:
            text = ''.join(fp.readlines())

        ex_preds, ex_lengths, ex_offsets = transformer.predict(text)

        if preds is None:
            preds = ex_preds
        else:
            preds = np.append(preds, ex_preds, axis=0)

        if output_folder is not None:
            # output the data to this folder as .pred files
            with open(output_folder / f'{f[:-4]}.pred', 'w') as fp:
                csvwriter = csv.writer(fp)
                # header
                csvwriter.writerow(output_header)
                for i in range(ex_preds.shape[0]):
                    start, stop = ex_offsets[i], ex_offsets[i] + ex_lengths[i]
                    entity = text[start:stop]
                    entity_type = transformer.label_id_map[np.argmax(
                        ex_preds[i, :]
                    )]
                    # do not save object entity types
                    if entity_type == 'O':
                        continue
                    row = [
                        f[:-4],
                        str(i + 1), start, stop, entity, entity_type, None
                    ]
                    csvwriter.writerow(row)
        lengths.append(ex_lengths)
        offsets.append(ex_offsets)

        # load in gold standard
        gs_fn = data_path / 'ann' / f'{f[:-4]}.gs'
        if args.label_transform:
            gs = label_transform(gs_fn)
        else:
            gs = load_labels(gs_fn)

        # convert labels to align with preds
        gs = sorted(gs, key=lambda x: x[1])
        label_tokens = ['O'] * len(ex_offsets)

        for i, g in enumerate(gs):
            idxStart = bisect_left(ex_offsets, g[1])
            idxStop = bisect_right(ex_offsets, g[1] + g[2])
            label_tokens[idxStart:idxStop] = [g[0]] * (idxStop - idxStart)

        # convert label tokens to label_ids
        # label_tokens = [label_to_id[pn_to_i2b2[l.upper()]] for l in label_tokens]
        if 'physionet_google' in args.data_dir:
            # HACK: map "NAME" gold standard to "PATIENT"
            label_tokens = [
                "PATIENT" if l.upper() == "NAME" else l for l in label_tokens
            ]

        label_tokens = [label_to_id[l.upper()] for l in label_tokens]
        labels.extend(label_tokens)

    # with open(args.output, 'wb') as fp:
    #     pickle.dump([files, preds, labels, lengths, offsets], fp)