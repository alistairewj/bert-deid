# Fixtures used by various tests
# Fixtures are a good way to
#  - load and share data across tests
#  - inject dependencies into tests
# We use environment variables in fixtures.
# If not present, we use default values.
import json
import os

import pytest
from bert_deid.model import Transformer


@pytest.fixture(scope="session")
def config_radiology_reports():
    """
    Load configuration file for the test radiology dataset.
    """
    # get path of conftest.py
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load the config json into a dictionary
    config_fn = os.path.join(
        dir_path, 'fake-data', 'config-radiology-reports.json'
    )

    # if config provided, update dict with values from config
    with open(config_fn, 'r') as fp:
        config = json.load(fp)

    return config


@pytest.fixture(scope="session")
def radiology_reports(config_radiology_reports):
    """
    Load the test radiology reports.
    """
    text_path = config_radiology_reports['text_path']

    reports_list = os.listdir(text_path)
    reports_list.sort()

    assert len(reports_list) == 7

    # load reports
    reports = {}
    for f in reports_list:
        with open(os.path.join(text_path, f), 'r') as fp:
            reports[f] = ''.join(fp.readlines())

    assert len(reports) == 7

    return reports


@pytest.fixture(scope="session")
def bert_i2b2_2014_model():
    """
    Load the model.
    """
    model_type = 'bert'  # -base-uncased'

    model_path = os.getenv('MODEL_PATH')
    if model_path is None:
        # use default
        model_path = '/data/models/bert-i2b2-2014'

    transformer = Transformer(
        model_type, model_path, max_seq_length=128, device='cpu'
    )
    return transformer
