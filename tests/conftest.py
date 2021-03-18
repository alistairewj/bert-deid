"""Fixtures used by various tests - primarily to load the model and example text."""
# Fixtures are a good way to
#  - load and share data across tests
#  - inject dependencies into tests
# We use environment variables in fixtures.
# If not present, we use default values.
import os
import pytest
from bert_deid.model.transformer import Transformer


@pytest.fixture(scope="session")
def ds_text():
    """Example text of a discharge summary."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'example_note.txt'), 'r') as fp:
        text = ''.join(fp.readlines())

    return text


@pytest.fixture(scope="session")
def ds_text_deid():
    """Deidentified text of a discharge summary."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'example_note_deid.txt'), 'r') as fp:
        text = ''.join(fp.readlines())

    return text


@pytest.fixture(scope="session")
def bert_i2b2_2014_model():
    """
    The BERT-deid model trained on the i2b2-2014 deid challenge corpus.
    """
    model_path = os.getenv('MODEL_DIR')
    if model_path is None:
        # use default
        model_path = '/data/models/bert-i2b2-2014'

    transformer = Transformer(model_path, device='cpu')
    return transformer
