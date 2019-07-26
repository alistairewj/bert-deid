# Fixtures used by various tests
# Fixtures are a good way to
#  - load and share data across tests
#  - inject dependencies into tests
import json
import os

import pytest
from bert_deid import model as bert_deid_model


@pytest.fixture(scope="session")
def config_radiology_reports():
    # get path of conftest.py
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load the config json into a dictionary
    config_fn = os.path.join(
        dir_path,
        'fake-data',
        'config-radiology-reports.json'
    )

    # if config provided, update dict with values from config
    with open(config_fn, 'r') as fp:
        config = json.load(fp)

    return config


@pytest.fixture(scope="session")
def radiology_reports(config_radiology_reports):
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
    return bert_deid_model.BertForDEID(
        model_dir="models/i2b2_2014"
    )
