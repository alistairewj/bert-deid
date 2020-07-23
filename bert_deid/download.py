import requests
import os
import logging

from google.cloud import storage

logger = logging.getLogger(__name__)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def download_model(model_dir=None):
    """
    Download the bert-deid-i2b2-2014 model to a given subfolder.
    """
    os.makedirs(model_dir, exist_ok=True)

    bucket_name = 'gs://bert-deid-1.0.0.physionet.org'
    files = [
        'added_tokens.json', 'config.json', 'label_set.bin',
        'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json',
        'training_args.bin', 'vocab.txt'
    ]
    logger.info(f'Beginning download of model files to {model_dir}')
    for fn in files:
        download_blob(bucket_name, f'model/{fn}', f'{model_dir}/{fn}')
        logger.info(f'Completed download for {fn}')
