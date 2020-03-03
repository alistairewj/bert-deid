from functools import partial
from bert_deid import processors


def test_simple_label_map():
    labels = [
    'NAME', 'DOCTOR', 'PATIENT', 'USERNAME', 'HCPNAME',
    'RELATIVEPROXYNAME', 'PTNAME', 'PTNAMEINITIAL',
    'PROFESSION',
    'LOCATION', 'HOSPITAL', 'ORGANIZATION', 'URL',
    'STREET', 'STATE',
    'CITY', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
    'PROTECTED_ENTITY', 'PROTECTED ENTITY',
    'NATIONALITY',
    'AGE', 'AGE_>_89', 'AGE > 89',
    'DATE', 'DATEYEAR',
    'BIOID', 'DEVICE', 'HEALTHPLAN',
    'IDNUM', 'MEDICALRECORD', 'ID', 'OTHER',
    'EMAIL', 'FAX', 'PHONE', 'CONTACT',
    'IPADDR', 'IPADDRESS'
    ]
    # emulate the format of the labels list:
    #   [entity_type, start, stop]
    labels = [[l] + [0, 1] for l in labels]

    L = len(labels)
    labels = processors.transform_label(labels, grouping='simple')

    assert len(labels) == L
    
    # check the name categories were mapped correctly
    for i in range(7):
        assert labels[i][0] == 'NAME'


def test_simple_label_map_bio():
    labels = [
        ['NAME', 0, 15, 'Antonio Vivaldi'],
        ['DOCTOR', 0, 3, 'Joe'],
        ['PROTECTED_ENTITY', 0, 21, 'Lower Blackrock Spire']
    ]

    # use bio decorator
    label_transform = partial(processors.transform_label, grouping='simple')
    label_transform = processors.bio_decorator(label_transform)

    labels = label_transform(labels)
    assert labels[0][0] == 'B-NAME'
    assert labels[1][0] == 'I-NAME'
    assert labels[2][0] == 'B-NAME'
    assert labels[3][0] == 'B-LOCATION'
    assert labels[4][0] == 'I-LOCATION'
    assert labels[5][0] == 'I-LOCATION'
