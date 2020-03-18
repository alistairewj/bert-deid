from functools import partial
from bert_deid import processors
from bert_deid.label import Label, LabelCollection, LABEL_MAP

PROCESSORS = processors.PROCESSORS


def test_simple_label_map():
    label_entities = [
        'NAME', 'DOCTOR', 'PATIENT', 'USERNAME', 'HCPNAME', 'RELATIVEPROXYNAME',
        'PTNAME', 'PTNAMEINITIAL', 'PROFESSION', 'LOCATION', 'HOSPITAL',
        'ORGANIZATION', 'URL', 'STREET', 'STATE', 'CITY', 'COUNTRY', 'ZIP',
        'LOCATION-OTHER', 'PROTECTED_ENTITY', 'PROTECTED ENTITY', 'NATIONALITY',
        'AGE', 'AGE_>_89', 'AGE > 89', 'DATE', 'DATEYEAR', 'BIOID', 'DEVICE',
        'HEALTHPLAN', 'IDNUM', 'MEDICALRECORD', 'ID', 'OTHER', 'EMAIL', 'FAX',
        'PHONE', 'CONTACT', 'IPADDR', 'IPADDRESS'
    ]
    labels = []
    for i, l in enumerate(label_entities):
        labels.append(Label(l, 0, 14, 'IGNORED ENTITY'))

    L = len(labels)

    # map each entity to standardized categories
    for label in labels:
        label.map_entity_type(LABEL_MAP['simple'])

    assert len(labels) == L

    # check the name categories were mapped correctly
    for i in range(7):
        assert labels[i].entity_type == 'NAME'


def test_simple_label_map_bio():
    label_set = LabelCollection('i2b2_2014', bio=True, transform='simple')
    label_set.from_list(
        [
            Label('NAME', 0, 15, 'Antonio Vivaldi'),
            Label('DOCTOR', 0, 3, 'Joe'),
            Label('PROTECTED_ENTITY', 0, 21, 'Lower Blackrock Spire')
        ]
    )

    assert label_set.labels[0].entity_type == 'B-NAME'
    assert label_set.labels[1].entity_type == 'I-NAME'
    assert label_set.labels[2].entity_type == 'B-NAME'
    assert label_set.labels[3].entity_type == 'B-LOCATION'
    assert label_set.labels[4].entity_type == 'I-LOCATION'
    assert label_set.labels[5].entity_type == 'I-LOCATION'


def test_processor():
    processor = processors.DeidProcessor('i2b2_2014', '/data/deid-gs/i2b2_2014')
    assert hasattr(processor, 'label_set')
    # num_labels = len(processor.get_labels())

    examples = processor.get_examples('train')
    assert len(examples) > 0
