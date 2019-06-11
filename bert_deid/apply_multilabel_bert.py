# De-identify radiology reports
import argparse
import os
import sys
import json
from datetime import datetime

from tqdm import tqdm

from bert_deid import bert_multilabel


parser = argparse.ArgumentParser(
    description='Apply multilabel bert to radiology reports')
parser.add_argument("--input",
                    default=None,
                    type=str,
                    required=True,
                    help=("The input data. Either a single text file, "
                          "or a folder containing .txt files"))
parser.add_argument("--model_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The directory where the model checkpoints are.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The directory where outputs should be placed.")


def main(args):
    args = parser.parse_args(args)

    # check we have the bert model we need
    if not os.path.exists(args.model_dir):
        raise ValueError((f'{args.model_dir} '
                          'subfolder with BERT model must exist.'))

    # check if the out directory exists; if not create it
    if not os.path.exists(args.output_dir):
        print('{} does not exist. Creating.'.format(args.output_dir))

        # call makedirs so we also make any intermediate directories needed
        os.makedirs(args.output_dir)

    files = os.listdir(args.input)
    N = len(files)

    # remove files without text extension, also remove text extension
    ext = '.txt'
    files = [x[0:-len(ext)] for x in files if x[-len(ext):] == ext]
    files.sort()

    # initialize bert
    bert_model = bert_multilabel.BertForRadReportLabel(
        model_dir=args.model_dir
    )

    print('{} - Parsing {} of {} files which have a text extension.'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M"), len(files), N))

    # create a dictionary of all labels for a single report
    preds = {}
    for i in tqdm(range(len(files))):
        rr_in_fn = os.path.join(args.input, files[i] + ext)
        out_fn = os.path.join(args.output_dir, files[i] + '.pred')

        with open(rr_in_fn, 'r') as fp:
            rr_text = ''.join(fp.readlines())

        # do not process if no report is present
        if len(rr_text) == 0:
            continue

        # get probabilities of each label and write out to file
        df_scores = bert_model.annotate(rr_text, document_id=files[i])
        df_scores.to_csv(out_fn, index=False)

        # convert to predictions - any sentence with positive label is positive
        for s in range(df_scores.shape[0]):
            pred = []
            for l in bert_model.labels:
                if df_scores.loc[s, l] > 0.5:
                    pred.append(l)
            preds[files[i] + f'.{s}'] = {}
            preds[files[i] + f'.{s}']['text'] = df_scores.loc[s, 'text']
            for p in pred:
                preds[files[i] + f'.{s}'][p] = 1
        #cols = bert_model.labels
        #idxTrue = (df_scores[cols].max(axis=0) > 0.5)

        #pred = df_scores[cols].columns.values[idxTrue]
        #pred = [x for x in pred if x != 'offset']
        #preds[files[i]] = ','.join(pred)

    out_fn = args.output_dir.rstrip('/') + '.json'
    with open(out_fn, 'w') as fp:
        fp.write(json.dumps(preds))

    now = datetime.now()
    print('{} - done!'.format(now.strftime("%Y-%m-%d %H:%M")))


if __name__ == '__main__':
    main(sys.argv[1:])
