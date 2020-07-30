import argparse
import os
import sys
import logging

from tqdm import tqdm
from bert_deid.model.transformer import Transformer
from bert_deid.download import download_model
from bert_deid.download import logger as download_logger

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if not default and envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault,
              self).__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description="bert-deid command line interface"
    )
    # shared arguments
    parser.add_argument(
        '--model_dir',
        action=EnvDefault,
        required=True,
        envvar='MODEL_DIR',
        help='Folder containing model to use'
    )
    parser.add_argument('--model_type', required=False, default='bert', help='')

    subparsers = parser.add_subparsers(dest="actions", title="actions")
    subparsers.required = True

    # sub-commands for the bert_deid interface

    # download the model
    apply = subparsers.add_parser(
        "download", help=("Download the model to the given model directory.")
    )

    # apply the model to some text
    apply = subparsers.add_parser(
        "apply", help="Apply bert-deid to text or text files"
    )
    apply.add_argument(
        '--text',
        nargs='?',
        type=str,
        default=(None if sys.stdin.isatty() else sys.stdin)
    )
    apply.add_argument(
        '--text_dir',
        type=str,
        default=None,
        help='De-identify all files in given folder.'
    )
    apply.add_argument('--repl', type=str, default='___')

    return parser.parse_args(arguments)


def apply(args):
    # if we have a files argument, we will run this over many files
    if args.text_dir is not None:
        if args.text is not None:
            raise ValueError(
                'Only one of text_dir/text arguments should be used.'
            )
        if not os.path.exists(args.text_dir):
            raise ValueError(
                f'Text directory given does not exist: {args.text_dir}'
            )
        if not os.path.isdir(args.text_dir):
            raise ValueError(
                f'Text directory is not a directory: {args.text_dir}'
            )

        file_list = os.listdir(args.text_dir)
        file_list = [
            os.path.join(args.text_dir, fn) for fn in file_list
            if os.path.isfile(os.path.join(args.text_dir, fn))
        ]
        if len(file_list) == 0:
            logger.warn(
                f'Directory provided did not contain files: {args.text_dir}'
            )
        # it's nice to process files in a sorted order
        file_list.sort()

    # load model in
    deid_model = Transformer(args.model_dir)

    # output will go to <filename>.deid
    if args.text_dir is not None:
        for fn in tqdm(file_list, total=len(file_list)):
            with open(fn, 'r') as fp:
                text = ''.join(fp.readlines())
            text_deid = deid_model.apply(text, args.repl)
            with open(f'{fn}.deid', 'w') as fp:
                fp.write(text_deid)
    else:
        print(deid_model.apply(args.text, args.repl), file=sys.stdout)


def download(args):
    download_model(args.model_dir)


def main(argv=sys.argv):
    # load in a trained model
    args = parse_arguments(argv[1:])

    if args.actions == 'download':
        # add logging if we call download from command line
        download_logger.setLevel(logging.INFO)
        download(args)
    elif args.actions == 'apply':
        apply(args)
    else:
        raise ValueError('Unrecognized action.')


if __name__ == '__main__':
    main()