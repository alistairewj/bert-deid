from __future__ import absolute_import, division, print_function

import json
import os
import logging
import csv
import argparse

from bert_deid.describe_data import harmonize_label
from bert_deid import model, utils
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig
from tqdm import tqdm
import torch
import pandas as pd
"""
Runs BERT deid on a set of text files.
Evaluates the output using gold standard annotations.

Optionally outputs mismatches to brat standoff format.
"""


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Config file
    parser.add_argument('-c', "--config",
                        default=None,
                        type=str,
                        help=("Configuration file (json)."
                              "Specifies folders and parameters."))
    # file to run
    parser.add_argument('-f', "--filename",
                        default=None,
                        required=True,
                        type=str,
                        help=("Filename to run bert deid on."))

    # If config is not provided, each argument can be given as an input
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        help="The directory with model weights/config files.")

    parser.add_argument("--text_path",
                        default=None,
                        type=str,
                        help=("The input data folder with individual"
                              " files to be deidentified."))
    parser.add_argument("--text_extension",
                        default='txt',
                        type=str,
                        help=("Extension for input files in input folder "
                              "(default: txt)"))

    # gold standard params
    parser.add_argument("--ref_path",
                        default=None,
                        type=str,
                        help=("The input ref folder with labels"
                              " for text files to be deidentified."))
    parser.add_argument("--ref_extension",
                        default='gs',
                        type=str,
                        help=("Extension for gold standard files in"
                              " ref folder (default: gs)"))

    parser.add_argument("--pred_orig_path",
                        default=None,
                        type=str,
                        help="Path to output unsimplified model labels.")
    parser.add_argument("--pred_path",
                        default=None,
                        type=str,
                        help="Path to output simplified model labels.")
    parser.add_argument("--pred_extension",
                        default='bert',
                        type=str,
                        help=("Extension for output labels"
                              " (default: bert)"))

    parser.add_argument('-b', '--brat_path', type=str,
                        default=None,
                        help='folder to output brat annotations')

    # parameters of the model/deid
    parser.add_argument("--task_name",
                        default='i2b2',
                        type=str,
                        choices=['i2b2', 'hipaa'],
                        help=("The name of the task to train. "
                              "Primarily defines the label set."))

    parser.add_argument('-m', '--method', type=str,
                        default='sentence',
                        choices=['sentence', 'overlap'],
                        help=('method for splitting text into'
                              ' individual examples.'))
    parser.add_argument('--step-size', type=int,
                        default=20,
                        help='if method="overlap", the token step size to use.')
    parser.add_argument('--sequence-length', type=int,
                        default=100,
                        help='if method="overlap", the max number of tokens per ex.')

    # Other parameters
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    # if config file is used, we ignore args and use config file/defaults
    # TODO: allow using environment variables?

    # prepare a dict with values from argparse (incl defaults)
    argparse_dict = vars(args)
    if args.config is not None:
        # if config provided, update dict with values from config
        with open(args.config, 'r') as fp:
            config = json.load(fp)

        argparse_dict.update(config)

    # if pred_orig_path exists, we are outputting non-simplified annot
    # if neither exist, warn the user that we are not saving ann to files
    if argparse_dict['pred_orig_path'] is None:
        if argparse_dict['pred_path'] is None:
            logger.info(
                '*** Not saving predictions to file as no pred folder given ***')

    # brat prep
    if argparse_dict['brat_path'] is not None:
        # ensure folder exists
        if not os.path.exists(argparse_dict['brat_path']):
            os.makedirs(argparse_dict['brat_path'])

        # add configuration files if needed to brat folder
        utils.add_brat_conf_files(argparse_dict['brat_path'])

    # ensure extension vars have the dot prefix
    for c in argparse_dict.keys():
        if c.endswith('_extension'):
            if argparse_dict[c][0] != '.':
                argparse_dict[c] = '.' + argparse_dict[c]

    input_ext = argparse_dict['text_extension']
    gs_ext = argparse_dict['ref_extension']
    pred_ext = argparse_dict['pred_extension']

    if argparse_dict['no_cuda']:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # load trained model
    output_model_file = os.path.join(argparse_dict['model_path'], WEIGHTS_NAME)
    output_config_file = os.path.join(argparse_dict['model_path'], CONFIG_NAME)
    if os.path.exists(output_model_file) & os.path.exists(output_config_file):
        # Load a trained model and config that you have fine-tuned
        bert_model = model.BertForDEID(
            model_dir=argparse_dict['model_path'],
            max_seq_length=argparse_dict['sequence_length'],
            token_step_size=argparse_dict['step_size']
        )
    else:
        raise ValueError('Folder %s did not have model and config file.',
                         argparse_dict['model_path'])

    bert_model.to(device)

    # check our file exists
    if os.path.exists(argparse_dict['text_path']):
        input_path = argparse_dict['text_path']
        input_fn = args.filename
        if not input_fn.endswith(input_ext):
            fn = input_fn
            input_fn += input_ext
        else:
            fn = input_fn[0:-len(input_ext)]
        input_fn = os.path.join(input_path, input_fn)
        if not os.path.exists(input_fn):
            raise ValueError('Input file %s does not exist.',
                             input_fn)
    else:
        raise ValueError('Input folder %s does not exist.',
                         argparse_dict['text_path'])

    logger.info("***** Running evaluation *****")

    # load the text
    with open(input_fn, 'r') as fp:
        text = ''.join(fp.readlines())

    # apply bert-deid
    df = bert_model.annotate(text, document_id=fn)

    logger.info("***** Initial annotations *****")
    logger.info("\n" + df.to_string())

    # pool the annotations
    if argparse_dict['sequence_length'] > argparse_dict['step_size']:
        df = bert_model.pool_annotations(df)
        logger.info("***** Pooling annotations *****")
        logger.info("\n" + df.to_string())

    # merges entity types + combines annotations <= 1 character apart
    df = utils.simplify_bert_ann(df, text, lowercase=True, dist=1)
    logger.info("***** Merging annotations *****")
    logger.info("\n" + df.to_string())

    # load ground truth
    gs_fn = os.path.join(argparse_dict['ref_path'], f'{fn}{gs_ext}')
    gs = pd.read_csv(gs_fn, header=0,
                     dtype={'entity': str,
                            'entity_type': str})

    # fix entities - lower case and group
    gs = utils.combine_entity_types(gs, lowercase=True)

    # reset annotation IDs for convenience
    gs['annotation_id'] = gs.index
    gs['annotation_id'] = gs['annotation_id'].map(lambda x: f'T{x}')

    # run comparison looking for exact/partial/misses
    cmp_ann = utils.compare_single_doc(gs, df)
    # add in the text/start/stop from gold standard annot
    cmp_ann = cmp_ann.merge(gs[['annotation_id',
                                'start', 'stop',
                                'entity_type', 'entity']],
                            how='left', on='annotation_id')

    logger.info("***** Performance comparison *****")
    logger.info("\n" + cmp_ann.to_string())


if __name__ == "__main__":
    main()
