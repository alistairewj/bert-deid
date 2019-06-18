import json
import sys
import argparse

from bert_deid import train_bert


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
    bert_arg_list = [
        'data_path', 'bert_model', 'task_name', 'model_path',
        'cache_dir', 'sequence_length', 'max_seq_length',
        'do_train', 'finetune',
        'do_eval', 'do_lower_case', 'train_batch_size',
        'eval_batch_size', 'learning_rate', 'num_train_epochs',
        'warmup_proportion', 'no_cuda', 'local_rank', 'seed',
        'gradient_accumulation_steps', 'fp16', 'loss_scale',
        'server_ip', 'server_port'
    ]

    bert_args = []
    do_train_passed = False
    for k in bert_arg_list:
        if k in config:
            if config[k] == 'True':
                bert_args.append(f'--{k}')
            elif config[k] == 'False':
                continue
            else:
                bert_args.append(f'--{k}')
                bert_args.append(str(config[k]))

            if k == 'do_train':
                do_train_passed = True

    if not do_train_passed:
        bert_args.append('--do_train')

    train_bert.main(bert_args)


if __name__ == "__main__":
    main(sys.argv[1:])
