import csv
import re
from functools import partial

LABEL_SETS = ('i2b2_2006', 'i2b2_2014', 'physionet', 'dernoncourt_lee', 'rr')

# Create a dictionary for mapping labels to standard categories.
LABEL_MEMBERSHIP = {
    'simple':
        [
            [
                'NAME',
                [
                    'NAME',
                    'DOCTOR',
                    'PATIENT',
                    'USERNAME',
                    'HCPNAME',
                    'RELATIVEPROXYNAME',
                    'PTNAME',
                    'PTNAMEINITIAL',
                    'KEYVALUE',
                ]
            ], ['PROFESSION', ['PROFESSION']],
            [
                'LOCATION',
                [
                    'LOCATION', 'HOSPITAL', 'ORGANIZATION', 'URL', 'STREET',
                    'STATE', 'CITY', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
                    'PROTECTED_ENTITY', 'PROTECTED ENTITY', 'NATIONALITY'
                ]
            ], ['AGE', ['AGE', 'AGE_>_89', 'AGE > 89']],
            ['DATE', ['DATE', 'DATEYEAR']],
            [
                'ID',
                [
                    'BIOID', 'DEVICE', 'HEALTHPLAN', 'IDNUM', 'MEDICALRECORD',
                    'ID', 'IDENTIFIER', 'OTHER'
                ]
            ],
            [
                'CONTACT',
                ['EMAIL', 'FAX', 'PHONE', 'CONTACT', 'IPADDR', 'IPADDRESS']
            ], ['O', ['O']]
        ],
    'hipaa':
        [
            [
                'NAME',
                [
                    'NAME',
                    'PATIENT',
                    'USERNAME',
                    'RELATIVEPROXYNAME',
                    'PTNAME',
                    'PTNAMEINITIAL',
                    'KEYVALUE',
                ]
            ],
            [
                'LOCATION',
                [
                    'LOCATION', 'ORGANIZATION', 'HOSPITAL', 'STREET', 'CITY',
                    'ZIP', 'URL', 'PROTECTED_ENTITY', 'PROTECTED ENTITY',
                    'LOCATION-OTHER'
                ]
            ],
            ['AGE', ['AGE', 'AGE_>_89', 'AGE > 89']],
            ['DATE', ['DATE', 'DATEYEAR']],
            [
                'ID',
                [
                    'BIOID', 'DEVICE', 'HEALTHPLAN', 'IDNUM', 'MEDICALRECORD',
                    'ID', 'IDENTIFIER', 'OTHER'
                ]
            ],
            [
                'CONTACT',
                [
                    'EMAIL',
                    'FAX',
                    'PHONE',
                    'CONTACT',
                    # it is unclear whether these are HIPAA in i2b2 paper
                    'IPADDR',
                    'IPADDRESS'
                ]
            ],
            [
                'O',
                [
                    'DOCTOR', 'HCPNAME', 'PROFESSION', 'STATE', 'COUNTRY',
                    'NATIONALITY', 'O'
                ]
            ]
        ],
    'binary':
        [
            [
                'PHI',
                [
                    'NAME',
                    'PATIENT',
                    'USERNAME',
                    'RELATIVEPROXYNAME',
                    'PTNAME',
                    'PTNAMEINITIAL',
                    'DOCTOR',
                    'KEYVALUE',
                    'HCPNAME',
                    'LOCATION',
                    'ORGANIZATION',
                    'HOSPITAL',
                    'PROTECTED_ENTITY',
                    'PROTECTED ENTITY',
                    'LOCATION-OTHER',
                    'STREET',
                    'CITY',
                    'ZIP',
                    'STATE',
                    'COUNTRY',
                    'NATIONALITY',
                    # two URLs in i2b2 which aren't URLs but web service names
                    'URL',
                    'AGE',
                    'AGE_>_89',
                    'AGE > 89',
                    'DATE',
                    'DATEYEAR',
                    'BIOID',
                    'DEVICE',
                    'HEALTHPLAN',
                    'IDNUM',
                    'MEDICALRECORD',
                    'ID',
                    'IDENTIFIER',
                    'OTHER',
                    'EMAIL',
                    'FAX',
                    'PHONE',
                    'CONTACT',
                    'PROFESSION',
                    'IPADDR',
                    'IPADDRESS'
                ]
            ],
            ['O', ['O']]
        ]
}

PYDEID_FEATURE2LABEL = {
    'age': 'AGE',
    'date': 'DATE',
    'email': 'CONTACT',
    'idnum': 'ID',
    'initials': 'NAME',
    'location': 'LOCATION',
    'mrn': 'ID',
    'name': 'NAME',
    'pager': 'ID',
    'ssn': 'ID',
    'telephone': 'CONTACT',
    'unit': 'ID',
    'url': 'CONTACT'
}


# convert the lists within each dictionary to a dict
def create_label_map(LABEL_MEMBERSHIP):
    LABEL_MAP = {}
    for grouping_name, label_members in LABEL_MEMBERSHIP.items():
        LABEL_MAP[grouping_name] = {}
        for harmonized_label, original_labels in label_members:
            for label in original_labels:
                LABEL_MAP[grouping_name][label] = harmonized_label

    return LABEL_MAP


LABEL_MAP = create_label_map(LABEL_MEMBERSHIP)


class Label(object):
    """Base class for a label.

    A label contains four attributes of primary interest:
        * entity_type - the type of entity which is labeled
        * start - the start offset of the label in the source text
        * length - the length of the label in the source text
        * entity - the actual text of the entity
    """
    def __init__(self, entity_type, start, length, entity):
        """Initialize a data processor with the location of the data."""
        self.entity_type = entity_type
        self.start = start
        self.length = length
        self.entity = entity

    def map_entity_type(self, mapping, force_upper=True):
        if force_upper:
            self.entity_type = mapping[self.entity_type.upper()]
        else:
            self.entity_type = mapping[self.entity_type]

    def contains(self, i):
        """Returns true if any label contains the offset."""
        return (self.start >= i) & ((self.start + self.length) < i)

    def overlaps(self, start, stop):
        """Returns true if any label contains the start/stop offset."""
        contains_start = (self.start >= start) & (self.start < stop)
        contains_stop = ((self.start + self.length) >=
                         start) & ((self.start + self.length) < stop)
        return contains_start | contains_stop


class LabelCollection(object):
    """Base class for a collection of labels.

    This class implements convenience functions such as label transformation.

    Args:
        bio (bool): transform to BIO format
        transform (str): how to group labels, options are:
            'simple' - high-level categories, e.g. one category for NAME
            'hipaa' - only categories covered by HIPAA
            'binary' - one label category: 'PHI' (non-PHI will become 'O')
    """
    def __init__(self, data_type, bio=None, transform=None):
        self.bio = bio
        self.transform = transform

        if data_type not in LABEL_SETS:
            raise NotImplementedError(
                (
                    f'Unrecognized data type {data_type}. '
                    f'Choices: {", ".join(LABEL_SETS)}'
                )
            )
        self.label_list = list(self.define_label_set(data_type))

        # update our label list using the transform
        # if self.transform is not None:
        #     self.label_list = list(
        #         set([LABEL_MAP[self.transform][l] for l in self.label_list])
        #     )
        #     self.label_list.sort()
        if self.transform is not None:
            self.label_list = list(
                l for l, _ in LABEL_MEMBERSHIP[self.transform]
            )
            self.label_list.sort()

        # enforce 'O' to be first label in all cases
        if 'O' in self.label_list:
            additional_label = ['O']
            self.label_list.pop(self.label_list.index('O'))
        else:
            additional_label = []

        if self.bio:
            # update label list to match BIO format
            self.label_list = list(self.label_list)

            self.label_list = ["B-" + l for l in self.label_list
                              ] + ["I-" + l for l in self.label_list]

        # add the 'O' back
        self.label_list = additional_label + self.label_list

        # now make label_list immutable
        self.label_list = tuple(self.label_list)
        self.labels = self.label_list

        # map labels to IDs
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        # name of this object - used in filename for caching
        self.__name__ = data_type
        if self.transform is None:
            self.__name__ += '_RawLabel'
        else:
            self.__name__ += f'_{self.transform}'

        if self.bio:
            self.__name__ += '_Bio'

    def from_csv(self, filename):
        """Loads annotations from a CSV file

        CSV file should have entity_type/start/stop columns."""
        with open(filename, 'r') as fp:
            csvreader = csv.reader(fp, delimiter=',', quotechar='"')
            header = next(csvreader)
            # identify which columns we want
            idx = [
                header.index('entity_type'),
                header.index('start'),
                header.index('stop'),
                header.index('entity')
            ]

            # iterate through the CSV and load in the labels
            self.labels = [
                Label(
                    entity_type=row[idx[0]],
                    start=int(row[idx[1]]),
                    length=int(row[idx[2]]) - int(row[idx[1]]),
                    entity=row[idx[3]]
                ) for row in csvreader
            ]

        # transform the labels as appropriate
        if len(self.labels) > 0:
            self.transform_labels()

    def from_list(self, labels):
        self.labels = labels
        for l in labels:
            if not isinstance(l, Label):
                raise ValueError(
                    'All elements of LabelCollection list must be type Label.'
                )

        # transform the labels if the user has requested it
        if len(self.labels) > 0:
            self.transform_labels()

    def sort_labels(self):
        if self.labels is not None:
            self.labels = sorted(self.labels, key=lambda x: x.start)

    def transform_label(self, label: Label):
        """Return the transformed version of an input label."""
        # creates a function to transform labels
        if self.transform is not None:
            label = label.map_entity_type(LABEL_MAP[self.transform])

        if self.bio:
            # convert label to BIO format
            if label.entity_type[:2] not in ("B-", "I-"):
                label = self.split_to_bio(label)

        return label

    def split_to_bio(self, label):
        # transform label into BIO scheme,
        # separate an entity on space and punctuation
        # e.g. labels = [('B-DATE', 36, 2),('I-DATE',39,1)]
        new_labels = []
        start = label.start
        split_by_space = label.entity.split(" ")
        is_first = True
        for each_split in split_by_space:
            split_by_punctuation = re.findall(
                r"\w+|[^\w\s]", each_split, re.UNICODE
            )
            for word in split_by_punctuation:
                # punctuation is treated as a distinct entity
                if is_first:
                    new_entity_type = "B-" + label.entity_type
                    is_first = False
                else:
                    new_entity_type = "I-" + label.entity_type
                new_labels.append(
                    Label(new_entity_type, start, len(word), entity=word)
                )
                start += len(word)
            start += 1

        return new_labels

    def transform_labels(self):
        """
        Groups entity types into a smaller set of categories.

        Modifies self.labels to use transformed entity types.
        """
        # creates a function to transform labels
        if self.transform is not None:
            for label in self.labels:
                label.map_entity_type(LABEL_MAP[self.transform])

        if self.bio:
            # convert labels to BIO format
            if self.labels[0].entity_type[:2] not in ("B-", "I-"):
                self.labels_to_bio()

    def labels_to_bio(self):
        # transform label into BIO scheme,
        # separate an entity on space and punctuation
        # e.g. labels = [('B-DATE', 36, 2),('I-DATE',39,1)]
        new_labels = []
        for label in self.labels:
            start = label.start
            split_by_space = label.entity.split(" ")
            is_first = True
            for each_split in split_by_space:
                split_by_punctuation = re.findall(
                    r"\w+|[^\w\s]", each_split, re.UNICODE
                )
                for word in split_by_punctuation:
                    # punctuation is treated as a distinct entity
                    if is_first:
                        new_entity_type = "B-" + label.entity_type
                        is_first = False
                    else:
                        new_entity_type = "I-" + label.entity_type
                    new_labels.append(
                        Label(new_entity_type, start, len(word), entity=word)
                    )
                    start += len(word)
                start += 1

        del self.labels
        self.labels = new_labels

    def define_label_set(self, label_type):
        if label_type == 'i2b2_2006':
            return (
                # non-entities
                'O',
                # names
                'DOCTOR',
                'PATIENT',
                # professions
                'PROFESSION',
                # locations
                'LOCATION',
                'HOSPITAL',
                # ages
                'AGE',
                # dates
                'DATE',
                # IDs
                'ID',
                # contacts
                'PHONE'
            )
        if label_type == 'i2b2_2014':
            return (
                # non-entities
                'O',
                # names
                'DOCTOR',
                'PATIENT',
                'USERNAME',
                # professions
                'PROFESSION',
                # locations
                'LOCATION',
                'HOSPITAL',
                'ORGANIZATION',
                'URL',
                'STREET',
                'STATE',
                'CITY',
                'COUNTRY',
                'ZIP',
                'LOCATION-OTHER',
                # ages
                'AGE',
                # dates
                'DATE',
                # IDs
                'BIOID',
                'DEVICE',
                'HEALTHPLAN',
                'IDNUM',
                'MEDICALRECORD',
                # contacts
                'EMAIL',
                'FAX',
                'PHONE'
            )
        if label_type == 'dernoncourt_lee':
            return (
                # non-entities
                'O',
                # names
                'DOCTOR',
                'PATIENT',
                # professions
                # locations
                'HOSPITAL',
                'STREET',
                'STATE',
                'COUNTRY',
                'ZIP',
                'LOCATION-OTHER',
                # ages
                'AGE',
                # dates
                'DATE',
                # IDs
                'IDNUM',
                # contacts
                'PHONE'
            )
        if label_type == 'physionet':
            return (
                # non-entities
                'O',
                # names
                'HCPNAME',
                'RELATIVEPROXYNAME',
                'PTNAME',
                'PTNAMEINITIAL',
                # professions
                # locations
                'LOCATION',
                # ages
                'AGE',
                # dates
                'DATE',
                'DATEYEAR',
                # IDs
                'OTHER',
                # contacts
                'PHONE'
            )

        if label_type == 'rr':
            return (
                # non-entities
                'O',
                # names
                'NAME',
                'KEYVALUE',
                # professions
                # locations
                'LOCATION',
                'PROTECTED_ENTITY',
                'ORGANIZATION',
                'NATIONALITY',
                # ages
                'AGE',
                'AGE_>_89',
                # dates
                'DATE',
                # IDs
                'IDENTIFIER',
                'OTHER',
                # contacts
                'CONTACT'
            )

        raise NotImplementedError('Unrecognized label_type.')
