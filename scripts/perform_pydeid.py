"""Use pydeid to identify location of PHI"""
import os
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import pkgutil
import pydeid

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

from pydeid.annotators import _patterns
from pydeid.annotators import Pattern
from pydeid.annotation import Document, EntityType
# load all modules on path
pkg = 'pydeid.annotators._patterns'
PATTERN_NAMES = [
    name for _, name, _ in pkgutil.iter_modules(_patterns.__path__)
]
PATTERN_NAMES.remove('_pattern')
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
        args.patterns = PATTERN_NAMES
    entity_type = [EntityType(pattern_name) for pattern_name in args.patterns]
    model = Pattern(args.patterns, entity_type)

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

        doc = Document(text)
        txt_annotated = model.annotate(doc)

        document_id = f'{f[:-4]}'
        annotation_df = pd.DataFrame(
            columns=[
                'document_id', 'annotation_id', 'start', 'stop', 'entity',
                'entity_type', 'comment'
            ]
        )
        for ann in txt_annotated.annotations:
            start, stop = ann.start, ann.end
            entity = ann.entity[0]
            entity_type = ann.entity_type
            annotation_df = annotation_df.append(
                {
                    "document_id": document_id,
                    "annotation_id": "",
                    "start": start,
                    "stop": stop,
                    "entity": entity,
                    "entity_type": entity_type,
                    "cooment": ""
                },
                ignore_index=True
            )

        if output_folder is not None:
            annotation_df.to_csv(output_folder / f'{f[:-4]}.pred', index=False)
