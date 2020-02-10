import csv 
import pandas as pd
import glob
import string
import argparse
import logging
from pathlib import Path
import os
import re
from tqdm import tqdm
from collections import OrderedDict

"""
Runs BERT deid on a set of text files.
Evaluates the output (matched correct PHI categories) using gold standard annotations.

Optionally outputs mismatches to brat standoff format.
"""

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def merge_token(data):
    # intervals: a list of (start row index, stop row index, merged string from start row to stop row)
    intervals = []
    sort_by_start = data.sort_values("start")
    for row_index in range(len(sort_by_start)-1):
        current_row = sort_by_start.iloc[row_index]
        next_row = sort_by_start.iloc[row_index+1]
        is_merged = False
        merge_with_next = False

        if current_row["stop"] == next_row["start"]:
            if next_row["entity"].strip(string.punctuation) == "" or current_row["entity_type"] == next_row["entity_type"]:
                # 1. stand-alone punctuation, disregard entity type conflict 
                #   i.e. "Cleveland Clinic." that has true label hospital but BERT predicts "." as "O"
                # 2. same entity type but tokenized into separate parts
                #   i.e. "8/16/2017" has true label DATE, BERT predicts into 5 parts: "8", "/", "16", ...
                merge_with_next = True
                if len(intervals) > 0:
                    if intervals[-1][-2] == row_index:
                        # merge prev, current, next predicted parts into one token
                        intervals[-1][-2] += 1
                        intervals[-1][-1] += next_row["entity"]
                        is_merged = True
                    
        if not is_merged and merge_with_next:
            intervals.append([row_index, row_index+1, current_row["entity"]+next_row["entity"]])
        elif not is_merged and not merge_with_next:
            merged_last_iteration = False
            if len(intervals) > 0:
                if intervals[-1][-2] >= row_index:
                    merged_last_iteration = True
            if not merged_last_iteration:
                intervals.append([row_index, row_index, current_row["entity"]])
            if row_index == len(sort_by_start)-2:
                # last token that does not merged with prev 
                intervals.append([row_index+1, row_index+1, next_row["entity"]])
            

    # create a new dataframe containing new start, stop, and merged string
    document_id = sort_by_start["document_id"].iloc[0]
    new_df = pd.DataFrame(columns=sort_by_start.columns)
    for start_row_index, stop_row_index, entity_str in intervals:
        entity_type = sort_by_start["entity_type"].iloc[start_row_index]
        start = sort_by_start["start"].iloc[start_row_index]
        stop = sort_by_start["stop"].iloc[stop_row_index]
        new_df = new_df.append({"document_id":document_id, "annotation_id":"","start":start,"stop":stop,
        "entity":entity_str, "entity_type":entity_type, "comment":""}, ignore_index=True)

    return new_df


def merge_BIO_pred(anno):
    sort_by_start = anno.sort_values("start")
    # find starting row index for each PHI predicted
    entity_starts = []
    for i, row in sort_by_start.iterrows():
        if row["entity_type"][0] == "B":
            entity_starts.append(i)
    # create a new dataframe merging BIO annotations to original ann
    new_anno = pd.DataFrame(columns=sort_by_start.columns)
    for i in range(len(entity_starts)):
        current_start_index = entity_starts[i]
        if i == len(entity_starts)-1:
            next_start_index = len(sort_by_start)
        else:
            next_start_index = entity_starts[i+1]
        entity_type = sort_by_start["entity_type"].iloc[current_start_index].split("-")[1]
        start = sort_by_start["start"].iloc[current_start_index]
        stop = sort_by_start["stop"].iloc[next_start_index-1]
        entity = ""
        for j in range(current_start_index, next_start_index-1):
            entity += str(sort_by_start["entity"].iloc[j])
            # HACK: this handles entity evaluation where BERT misses
            # middle of entity but predict start and end correct
            # in such case, token evaluation is worst than entity evaluation
            # if only cares about correctly predict START, STOP, ENTITY TYPE.
            for _ in range(sort_by_start["stop"].iloc[j], sort_by_start["start"].iloc[j+1]):
                entity += " "
        entity += str(sort_by_start["entity"].iloc[next_start_index-1])
        new_anno = new_anno.append({"document_id":sort_by_start["document_id"].iloc[0], "annotation_id":"",
        "start":start,"stop":stop,"entity":entity, "entity_type":entity_type,"cooment":""}, ignore_index=True)

    return new_anno

def get_entities(data):
    entities = [(data["entity_type"].iloc[i].upper(), data["start"].iloc[i],
    data["stop"].iloc[i], data["entity"].iloc[i].upper()) for i in range(len(data))]


    return entities

def get_tokens(data):
    tokens = []
    # break up into word-level token
    for _, row in data.iterrows():
        entities = str(row["entity"]).split(" ")
        entity_type = row["entity_type"]
        start = row["start"]
        for token in entities:
            if len(token) > 0:
                tokens.append((entity_type.upper(), start, start+len(token), token.upper()))
            start += len(token) + 1
    return tokens

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path", required=True, type=str, help="Path for predicted albels"
    )

    parser.add_argument(
        "--text_path",required=True,type=str,help="Path for de-identified text"
    )

    parser.add_argument(
        "--ref_path", required=True, type=str, help="Gold standard labels"
    )

    parser.add_argument(
        "--pred_extension",default="pred",type=str,
        help=("Extension for prediction labels (default: pred)")
    )

    parser.add_argument(
        "--ref_extension",default="gs",type=str,
        help=("Extension for gold standard files in ref folder (default: gs)")
    )

    parser.add_argument(
        "--label_transform", action="store_true",
        help="Whether labels are transformed using BIO"
    )

    parser.add_argument(
        "--token_eval", action="store_true",
        help="Whether to perform token evaluation"
    )

    parser.add_argument(
        "--entity_eval", action="store_true",
        help="Whether to perform entity evaluation, only allowed if label is transformed with BIO scheme"
    )

    parser.add_argument(
        "--log",type=str,default=None,help="text file to output false positive/negative to"
    )

    parser.add_argument(
        "--csv_path",type=str,default=None,help="CSV file to output errors for labelling"
    )

    args = parser.parse_args()

    ref_path = Path(args.ref_path)
    pred_path = Path(args.pred_path)

    csv_path = None
    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
        if not os.path.exists(csv_path.parents[0]):
            os.makedirs(csv_path.parents[0])

    log_path = None
    if args.log is not None:
        log_path = Path(args.log)
        if not os.path.exists(log_path.parents[0]):
            os.makedirs(log_path.parents[0])

        log_text = OrderedDict(
            [["False Negatives", ''],["False Positives",'']]
        )

    input_ext = '.txt'
    gs_ext = args.ref_extension
    if not gs_ext.startswith('.'):
        gs_ext = '.' + gs_ext
    pred_ext = args.pred_extension
    if not pred_ext.startswith('.'):
        pred_ext = '.' + pred_ext

    # read files from folder
    if os.path.exists(args.text_path):
        input_path = Path(args.text_path)
        files = os.listdir(input_path)

        # remove extension and get file list
        input_files = set(
            [f[0:-len(input_ext)] for f in files if f.endswith(input_ext)]
        )

        input_files = sorted(list(input_files))
    else:
        raise ValueError('Input folder %s does not exist.', args.text_path)
    
    if args.token_eval and not args.entity_eval:
        logger.info("***** Running token evaluation *****")
    elif args.entity_eval and not args.token_eval and args.label_transform:
        logger.info("***** Running entity evaluation *****")
    else: 
        raise ValueError (
                "Invalid input arguments. Make sure labels are transformed if performing entity evaluation"
        )
    logger.info(" Num examples = %d", len(input_files))

    true_positive, false_positive, false_negative = 0,0,0
    perf_all = {}
    total_eval = 0
    for fn in tqdm(input_files,total=len(input_files)):
        # load the text
        with open(input_path / f'{fn}{input_ext}', 'r') as fp:
            text = ''.join(fp.readlines())

        # load output of bert-deid
        fn_pred = pred_path / f'{fn}{pred_ext}'
        df = pd.read_csv(
            fn_pred, header=0,delimiter=",",
            dtype={
                'entity':str, 'entity_type':str
                })
        # load ground truth
        gs_fn = ref_path / f'{fn}{gs_ext}'
        gs = pd.read_csv(
            gs_fn, header=0, delimiter=",",
            dtype={
                'entity':str,'entity_type':str
            }
        )

        if args.label_transform:
            df = merge_BIO_pred(df)


        # entity evaluation
        if args.entity_eval: 
            pred = get_entities(df) 
            true = get_entities(gs)
        
        else:  # token evaluaiton
            if len(df) > 1 and not args.label_transform:
                # at least two predicted to be merged
                df = merge_token(df)
            pred = get_tokens(df)
            true = get_tokens(gs)


        total_eval += len(true)
        current_tp = len(set(pred) & set(true))
        current_fp = len(pred) - current_tp
        current_fn = len(true) - current_tp
        true_positive += current_tp
        false_positive += current_fp
        false_negative += current_fn

        if (log_path is not None) and ((current_fp > 0) or (current_fn > 0)):
            for key in log_text.keys():
                if key == 'False Positives':
                    false_set = set(pred).difference((set(pred) & set(true)))
                else: 
                    false_set = set(true).difference((set(pred) & set(true)))
                sorted_fp = sorted(list(false_set), key=lambda x: x[1])
                for (entity_type, start, stop, entity) in sorted_fp:
                    log_text[key] += f'{fn},'
                    log_text[key] += text[max(start - 50, 0):start].replace('\n', ' ')
                    entity = text[start:stop]
                    log_text[key] += "**" + entity.replace('\n', ' ') + "**"
                    log_text[key] += text[stop:min(stop+50, len(text))].replace("\n", " ")
                    log_text[key] += "\n"
                    if (',' in entity) or ("\n" in entity) or ('"' in entity):
                        entity = '"' + entity.replace('"', '""') + '"'
                    log_text[key] += f'{fn},,{start},{stop},{entity},{entity_type},\n'
        
        perf_all[fn] = {'tp':current_tp, 'fp':current_fp, 'fn':current_fn}
    
    # convert to dataframe
    info_df = pd.DataFrame.from_dict(perf_all, orient='index')

    # print (info_df)
    if args.token_eval:
        print ("Number of tokens: {}".format(total_eval))
    else:
        print ("Number of entities: {}".format(total_eval))
    print ("True positives: {}".format(true_positive))
    print ("False positives: {}, false negatives: {}".format(false_positive, false_negative))

    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    f1 = 2*precision*recall / (precision + recall)
    print (f'Micro Se: {recall:0.3f}')
    print (f'Micro P+: {precision:0.3f}')
    print (f'Micro F1: {f1:0.3f}')

    if log_path is not None:
        # overwrite the log file
        with open(log_path, 'w') as f:
            for k, txt in log_text.items():
                f.write(f'==={k} ===\n{txt}\n\n')

    if csv_path is not None:
        info_df.to_csv(csv_path)


if __name__ == "__main__":
    main()


    


