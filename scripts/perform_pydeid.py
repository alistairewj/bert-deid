"""Use pydeid to identify location of PHI"""
import os
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import pkgutil 
import pydeid
from pydeid import annotator

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# load all modules on path
pkg = 'pydeid.annotator._patterns'
PATTERN_NAMES = [name for _, name, _ in pkgutil.iter_modules(
    pydeid.annotator._patterns.__path__
)]
_PATTERN_NAMES = PATTERN_NAMES + ['all']

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
        "--feature",
        type=str,
        nargs='+',
        default=None,
        choices=_PATTERN_NAMES,
        help="Perform rule-based approach with pydeid patterns: "
            f"{', '.join(_PATTERN_NAMES)}"
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Output folder pydeid result",
    )
    args = parser.parse_args()

    args.patterns = []
    if args.feature is not None:
        for f in args.feature:
            f = f.lower()
            if f not in _PATTERN_NAMES:
                raise ValueError("Invalid feature name")
            args.patterns.append(f)

    if 'all' in args.patterns:
        modules = PATTERN_NAMES
    else:
        modules = args.patterns
    
    model = annotator.Pattern(modules)

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

    files = os.listdir(data_path / 'txt')
    files = [f for f in files if f.endswith('.txt')]
    data = []

    preds = None
    lengths = []
    offsets = []
    labels = []

    for f in tqdm(files, total=len(files), desc='Files'):
        with open(data_path / 'txt' / f, 'r') as fp:
            text = ''.join(fp.readlines())

        ex_preds = model.annotate(data=text, document_id=f'{f[:-4]}')


        if output_folder is not None:
            ex_preds.to_csv(output_folder / f'{f[:-4]}.pred', index=False)
