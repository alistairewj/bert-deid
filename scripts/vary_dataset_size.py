import json
import sys
import os
import argparse

from bert_deid import train_bert
from bert_deid import create_csv


def arg_dict_to_list(arg_dict):
    arg_list = []
    for c in arg_dict:
        if type(arg_dict[c]) is bool:
            if arg_dict[c]:
                arg_list.append(f'--{c}')
        else:
            arg_list.append(f'--{c}')
            arg_list.append('{}'.format(arg_dict[c]))
    return arg_list


def main(args):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('-c', "--config",
                        required=True,
                        type=str,
                        help=("Configuration file (json)."
                              "Specifies folders and parameters."))

    args = parser.parse_args(args)

    # if config provided, update dict with values from config
    with open(args.config, 'r') as fp:
        config = json.load(fp)

    # hard-coded list of parameters for bert training
    csv_arg_names = [
        'input_path', 'text_extension', 'ref_extension',
        'data_path', 'task_name', 'group_tags',
        'method', 'step_size', 'sequence_length',
        'bert_model', 'quiet'
    ]

    create_csv_args = {}
    for k in csv_arg_names:
        if k in config:
            if config[k] == 'True':
                create_csv_args[k] = True
            elif config[k] == 'False':
                continue
            else:
                create_csv_args[k] = str(config[k])

    # adding an input_test_path arg lets us generate test.csv
    if 'input_test_path' in config:
        print('Generating test set using {}'.format(
            config['input_test_path']
        ))
        create_csv_args['input_path'] = config['input_test_path']
        csv_arg_list = arg_dict_to_list(create_csv_args)
        create_csv.main(csv_arg_list)
    # reset the input_path for training set generation later
    create_csv_args['input_path'] = config['input_path']

    # hard-coded list of parameters for bert training
    bert_arg_names = [
        'data_path', 'bert_model', 'task_name', 'model_path',
        'cache_dir', 'max_seq_length',
        'do_train', 'finetune',
        'do_eval', 'do_lower_case', 'train_batch_size',
        'eval_batch_size', 'learning_rate', 'num_train_epochs',
        'warmup_proportion', 'no_cuda', 'local_rank', 'seed',
        'gradient_accumulation_steps', 'fp16', 'loss_scale',
        'server_ip', 'server_port'
    ]

    bert_args = {}
    for k in bert_arg_names:
        if k in config:
            if config[k] == 'True':
                bert_args[k] = True
            elif config[k] == 'False':
                continue
            else:
                bert_args[k] = str(config[k])

    # we force do_train and do_eval to be true regardless of args
    bert_args['do_train'] = True
    bert_args['do_eval'] = True
    # below outputs args to model_path/eval_predictions.csv
    bert_args['output_predictions'] = True

    # we use 10%, 20%, ... 100% of data for training
    for d in range(10, 101, 10):
        # save model/outputs in subdirectory with percentage as the subdir name
        subset = f'{d:d}'
        paths_update = {
            'data_path': os.path.join(config['data_path'], subset),
            'model_path': os.path.join(config['model_path'], subset),
            'pred_path': os.path.join(config['pred_path'], subset),
            'pred_orig_path': os.path.join(config['pred_orig_path'], subset)
        }
        for c in paths_update:
            if c in csv_arg_names:
                create_csv_args[c] = paths_update[c]
            if c in bert_arg_names:
                bert_args[c] = paths_update[c]

        print(f'Generating CSV for {d}% of data')
        csv_arg_list = arg_dict_to_list(create_csv_args)
        create_csv.main(csv_arg_list)

        print('Training model.')
        bert_arg_list = arg_dict_to_list(bert_args)
        train_bert.main(bert_arg_list)


if __name__ == "__main__":
    main(sys.argv[1:])
