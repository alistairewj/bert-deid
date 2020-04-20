"""Combine BERT results with pydeid results"""
import os
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

logger = logging.getLogger()
logger.setLevel(logging.WARNING)



def combine_results(bert_df, pydeid_df, text):
    
    def get_intervals(df):
        arr = []
        for i, row in df.iterrows():
            arr.append((row['start'], row['stop']))
        
        return arr
    
    bert_intervals = get_intervals(bert_df)
    pydeid_intervals = get_intervals(pydeid_df)
    
    combined_intervals = bert_intervals + pydeid_intervals
            
    
    def merge_intervals(intervals): 

        sorted_intervals = sorted(intervals, key=lambda tup: tup[0])

        if not sorted_intervals:  # no intervals to merge
            return

        # low and high represent the bounds of the current run of merges
        low, high = sorted_intervals[0]

        for iv in sorted_intervals[1:]:
            if iv[0] <= high:  # new interval overlaps current run
                high = max(high, iv[1])  # merge with the current run
            else:  # current run is over
                yield low, high  # yield accumulated interval
                low, high = iv  # start new run

        yield low, high
    
    combined_intervals = merge_intervals(combined_intervals)
    
    new_df = pd.DataFrame(columns=bert_df.columns)
    if len(bert_df) > 0:
        document_id = bert_df['document_id'].iloc[0]
    elif len(pydeid_df) > 0:
        document_id = pydeid_df['document_id'].iloc[0]
    else:
        return new_df
    for start, stop in merge_intervals(combined_intervals):
        entity = text[start:stop]
        # only takes care of binary 
        entity_type = 'PHI'
        new_df = new_df.append({"document_id":document_id, "annotation_id":"",
        "start":start,"stop":stop,"entity":entity, "entity_type":entity_type,"cooment":""}, ignore_index=True)
        
    return new_df


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
        "--bert_result_dir",
        default='/enc_data/pred/bert-i2b2-2014',
        type=str,
        required=True,
        help="BERT annotation result dir.",
    )
    parser.add_argument(
        "--pydeid_result_dir", 
        default='/enc_data/pred/pydeid-i2b2-2014', 
        type=str, 
        required=True,
        help="Pydeid annotation result dir"
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Output folder combined result",
    )
    args = parser.parse_args()

    bert_result_path = Path(args.bert_result_dir)
    pydeid_result_path = Path(args.pydeid_result_dir)
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

    bert_result_files = os.listdir(bert_result_path)
    bert_result_files = [f for f in bert_result_files if f.endswith('.pred')]
    data = []

    preds = None
    labels = []

    for fn in tqdm(bert_result_files,total=len(bert_result_files)):

        # load output of bert-deid
        bert_fn_pred = bert_result_path / f'{fn}'
        bert_df = pd.read_csv(
            bert_fn_pred, header=0,delimiter=",",
            dtype={
                'entity':str, 'entity_type':str
                })
        pydeid_fn_pred = pydeid_result_path / f'{fn}'
        pydeid_df = pd.read_csv(
            pydeid_fn_pred, header=0,delimiter=",",
            dtype={
                'entity':str, 'entity_type':str
            })


        with open(data_path / 'txt' / f'{fn[:-4]}txt', 'r') as fp:
            text = ''.join(fp.readlines())

        ex_preds = combine_results(bert_df, pydeid_df, text)
        if output_folder is not None:
            ex_preds.to_csv(output_folder / f'{fn[:-4]}pred', index=False)


