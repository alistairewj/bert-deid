import json
import sys
import argparse

from bert_deid import create_csv


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
    arg_name_list = [
        'input_path', 'text_extension', 'ref_extension',
        'data_path', 'task_name', 'group_tags',
        'method', 'step_size', 'sequence_length',
        'bert_model', 'quiet'
    ]

    arg_list = []
    for k in arg_name_list:
        if k in config:
            if config[k] == 'True':
                arg_list.append(f'--{k}')
            elif config[k] == 'False':
                continue
            else:
                arg_list.append(f'--{k}')
                arg_list.append(str(config[k]))

    create_csv.main(arg_list)

    # adding an input_test_path arg also lets us generate test.csv
    if 'input_test_path' in config:
        for i, a in enumerate(arg_list):
            if a == '--input_path':
                # also generate test csv since the arg was passed
                print('Also generating test set using {}'.format(
                    config['input_test_path']
                ))
                arg_list[i+1] = config['input_test_path']
                create_csv.main(arg_list)
                break


if __name__ == "__main__":
    main(sys.argv[1:])
