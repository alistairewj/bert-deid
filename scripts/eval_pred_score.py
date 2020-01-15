import csv 
import pandas as pd
import glob
import string

# maps dataframe id to corresponding filename
df_id_filename = {}

def load_data(path_to_anno, path_to_gold, anno_ext, gold_ext):
    """
    Parameter: path to annotated and gold standard files and corresponding externsion
    Return: (list of dataframes about annotated data, list of dataframes about true data)
    """
    anno_dfs, gold_dfs = [], []
    gold_files = glob.glob(path_to_gold+"*"+gold_ext)
    counter = 0
    for gold_file in gold_files:
        gold_dfs.append(pd.read_csv(gold_file,delimiter=",",header=0))
        anno_file = path_to_anno+gold_file.split("/")[-1].split(".")[0] + anno_ext
        anno_dfs.append(pd.read_csv(anno_file,delimiter=",",header=0))
        df_id_filename[counter] = gold_file.split("/")[-1]
        counter += 1
    
    assert (len(anno_dfs) == len(gold_dfs))
    assert (len(df_id_filename) == len(anno_dfs))
    return anno_dfs, gold_dfs


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

    # print (new_df)
    return new_df

def evaluate(annos, golds, evaluation_type):
    """
    parameters:
        anno, gold: a list of dataframes
        evaluation_type: string, "ENTITY" or "TOKEN"
    return:
        tp, fp, fn
    """
    def get_entities(data):
        entities = [(data["entity_type"].iloc[i].upper(), data["start"].iloc[i],
        data["stop"].iloc[i]) for i in range(len(data))]
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
                    tokens.append((entity_type.upper(), start, start+len(token)))
                start += len(token) + 1
        return tokens

    
    tp,fp,fn = 0, 0, 0
    for i in range(len(golds)):
        if evaluation_type.upper() == "ENTITY":
            # entity level
            pred = get_entities(annos[i])
            true = get_entities(golds[i])
        elif evaluation_type.upper() == "TOKEN":
            # token level 
            if len(annos[i]) > 1:
                # at least two predicted to be merged
                annos[i] = merge_token(annos[i])
            pred = get_tokens(annos[i])
            true = get_tokens(golds[i])
        else:
            print ("Invalid input for evaluation type.")
            return 
        current_tp = len(set(pred) & set(true))
        tp += current_tp
        fp += len(pred) - current_tp
        fn += len(true) - current_tp

        ## Comment out to see text results for false positive and false negative
        # if current_tp < len(pred):
        #     fp_set = set(pred).difference((set(pred) & set(true)))
        #     sorted_fp = sorted(list(fp_set), key=lambda x: x[1])
        #     print ("document {}, false positive: ".format(df_id_filename[i]))
        #     for each in sorted_fp:
        #         print (str(each))
        # if current_tp < len(true):
        #     fn_set = set(true).difference((set(true) & set(pred)))
        #     sorted_fn = sorted(list(fn_set), key=lambda x:x[1])
        #     print ("document {}, false negative: ".format(df_id_filename[i]))
        #     for other in sorted_fn:
        #         print (str(other))
        
        
    print ("TP:{}, FP:{}, FN:{}".format(tp,fp,fn))
    print ("Precision: {} and Recall: {}".format(tp/(tp+fp), tp/(tp+fn)))
    return tp, fp, fn


# fake data
# doc 1 includes 
    # 1. match (TP), 
    # 2. system misses an entity (FN) 
    # 3. system assigns wrong entity type (FP and FN)
    # 4. system misses end boundary of entity (FP and FN)
# doc 2 includes 
    # 1. system hypothesized an entity (FP)
    # 2. system predicts two different entities as one 
    # 3. system extends both boundaries of an entity 
# doc 3 includes
    # 1. miss one token in an entity
    # 2. system gets boundaries and entity type wrong
    # 3. system breaks up entity into token (token level: TP)

# path_to_anno = "/home/jingglin/research/fake_data/pred/"
# path_to_gold = "/home/jingglin/research/fake_data/true/"
# anno_dfs, gold_dfs = load_data(path_to_anno, path_to_gold, ".phi", ".gs")

# # entity level 
# assert (evaluate(anno_dfs, gold_dfs, "ENTITY") == (2,10,9))

# # token level
# assert (evaluate(anno_dfs, gold_dfs, "TOKEN") == (14,13,9))

path_to_anno = "/home/jingglin/research/data/pred/physionet/train/"
path_to_gold = "/home/jingglin/research/deid-gs/physionet/train/ann/"
anno_dfs, gold_dfs = load_data(path_to_anno, path_to_gold, ".pred", ".gs")
evaluate(anno_dfs, gold_dfs, "TOKEN")

# individual file
# anno_df = pd.read_csv("/home/jingglin/research/data/pred/physionet/6-1.pred", delimiter=",", header=0)
# gold_df = pd.read_csv("/home/jingglin/research/deid-gs/physionet/test/ann/6-1.gs", delimiter=",", header=0)

# evaluate([anno_df], [gold_df], "ENTITY")


    


