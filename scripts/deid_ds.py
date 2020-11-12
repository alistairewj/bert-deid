import argparse
import json
import os

from tqdm import tqdm
import pandas as pd

from bert_deid.model.transformer import Transformer


def main():
    parser = argparse.ArgumentParser(
        description='De-identify discharge summaries'
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='scripts/icu_ds.csv',
        help='path to discharge summary CSV'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./ds-bert/',
        help='folder to output annotation files'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='/home/alistairewj/bert-model-i2b2-2014',
        help='folder containing model'
    )
    parser.add_argument(
        '-s',
        '--start',
        type=int,
        default=0,
        help='index of first discharge summary row to de-identify'
    )
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        default=61997,
        help='number of discharge summary rows to de-identify'
    )
    args = parser.parse_args()

    model_path = args.model
    transformer = Transformer(model_path, device='cpu')

    notes = pd.read_csv(args.input)
    notes.sort_values('row_id', inplace=True)

    if args.start > 0:
        notes = notes.iloc[args.start:]

    if args.number < notes.shape[0]:
        notes = notes.iloc[:args.number]

    output_path = args.output
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i, row in tqdm(notes.iterrows(), total=notes.shape[0]):
        text = row['text']
        labels = transformer.predict(text)
        labels = pd.DataFrame(
            labels, columns=['probability', 'entity_type', 'offset', 'length']
        )
        labels.sort_values(['offset', 'length', 'entity_type'], inplace=True)
        labels.to_csv(
            os.path.join(output_path, f'{row["row_id"]}.bert'),
            header=True,
            index=False
        )


if __name__ == '__main__':
    main()
