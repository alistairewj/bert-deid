import re
import csv
import os
import sys
import logging

# from nltk.tokenize import sent_tokenize
from bert_deid.utils import create_hamonize_label_dict

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_labels(fn):
    """Loads annotations from a CSV file with entity_type/start/stop columns."""
    with open(fn, 'r') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        header = next(csvreader)
        # identify which columns we want
        idx = [
            header.index('entity_type'),
            header.index('start'),
            header.index('stop')
        ]

        # iterate through the CSV and load in the labels as a list of tuples
        #  (label name, start index, length of entity)
        # e.g. labels = [('DATE', 36, 5), ('AGE', 45, 2)]
        labels = [
            (
                row[idx[0]], int(row[idx[1]]),
                int(row[idx[2]]) - int(row[idx[1]])
            ) for row in csvreader
        ]
    
    return labels

def load_labels_string(fn):
    """Loads annotations from a CSV file with entity_type/start/stop columns."""
    with open(fn, 'r') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        header = next(csvreader)
        # identify which columns we want
        idx = [
            header.index('entity_type'),
            header.index('start'),
            header.index('stop'), 
            header.index('entity')
        ]

        # iterate through the CSV and load in the labels as a list of tuples
        #  (label name, start index, length of entity)
        # e.g. labels = [('DATE', 36, 5, 'entity'), ('AGE', 45, 2, 'entity')]
        labels = [
            (
                row[idx[0]], int(row[idx[1]]),
                int(row[idx[2]]) - int(row[idx[1]]), row[idx[3]]
            ) for row in csvreader
        ]
    
    return labels

def label_transform(fn):
    # transform label into BIO scheme, separate an entity on space and punctuation
    # e.g. labels = [('B-DATE', 36, 2),('I-DATE',39,1)]
    labels = load_labels_string(fn)
    new_labels = []
    for (entity_type, start, _, entity) in labels:
        split_by_space = entity.split(" ")
        is_first = True
        for each_split in split_by_space:
            split_by_punctuation = re.findall(r"\w+|[^\w\s]", each_split, re.UNICODE)
            for word in split_by_punctuation:
                if is_first:
                    new_entity_type = "B-"+entity_type
                    is_first = False
                else:
                    new_entity_type = "I-"+entity_type
                new_labels.append((new_entity_type, start, len(word)))
                start += len(word)
            start += 1

    return new_labels


class InputExample(object):
    """A single training/test example."""
    def __init__(self, guid, text, labels=None):
        """Constructs an InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The text comprising the sequence.
            label: (Optional) list. Stand-off labels. List of lists.
                Each element has the form:
                (label_name, start_offset, length)
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class DataProcessor(object):
    """Base class for data converters."""
    def __init__(self, data_dir, label_transform=False):
        """Initialize a data processor with the location of the data."""
        self.data_dir = data_dir
        self.data_filenames = None
        self.label_list = None
        self.label_transform = label_transform

    def get_labels(self):
        """Gets the list of labels for this data set."""
        if self.label_list is None:
            raise NotImplementedError()
        if self.label_transform:
            return ['O'] + ["B-"+l for l in self.label_list if l != 'O'] + ["I-"+l for l in self.label_list if l != 'O'] 
        else:
            return list(self.label_list)

    def _create_examples(self, fn, mode):
        raise NotImplementedError()

    def get_examples(self, mode):
        if mode not in ('train', 'test', 'val'):
            raise ValueError(
                (
                    f'Incorrect mode="{mode}", '
                    'expected "train", "test", or "val".'
                )
            )
        if self.data_filenames is None:
            # classes which inherit should define self.data_filenames dictionary
            raise NotImplementedError
        elif mode not in self.data_filenames:
            raise ValueError(
                f'{mode} is not available for {self.__class__.__name__}'
            )

        fn = os.path.join(self.data_dir, self.data_filenames[mode])
        return self._create_examples(fn, mode)

    def _read_file(self, input_file, delimiter=',', quotechar='"'):
        """Reads a comma separated value file."""
        fn = os.path.join(self.data_dir, input_file)
        with open(fn, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return self._read_file(input_file, delimiter='\t', quotechar=quotechar)

    def _read_csv(self, input_file, quotechar='"'):
        """Reads a comma separated value file."""
        return self._read_file(input_file, delimiter=',', quotechar=quotechar)


class CoNLLProcessor(DataProcessor):
    """Processor for the gold standard de-id data set."""
    def __init__(self, data_dir, label_transform=False):
        """Initialize a data processor with the location of the data."""
        super().__init__(data_dir, label_transform=label_transform)
        # conll2003 filenames
        self.data_filenames = {
            'train': 'train.txt',
            'test': 'test.txt',
            'val': 'valid.txt'
        }
        self.label_list = (
            'O', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC',
            'I-ORG', 'I-PER'
        )

    # def _create_examples(self, lines, set_type):
    def _create_examples(self, fn, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        sentence = []
        label = []
        # n keeps track of the number of samples processed
        n = 0

        with open(fn, encoding="utf-8") as fp:
            lines = [x.rstrip('\n') for x in fp.readlines()]

        # guid = [dataset - sample number - starting line number]
        guid = "%s-%s-%s" % (set_type, n, 2)
        for (i, line) in enumerate(lines):
            if len(line) == 0:
                if len(sentence) == 0:
                    continue
                # end of sentence

                # reformat labels to be a list of anns: [start, stop, entity]
                # e.g. [[0, 2, 'O'], [2, 6, 'ORG], ...]
                label_offsets = []
                s_len = 0
                for j, l in enumerate(sentence):
                    # label_new = combine_labels[label[j]]
                    if self.label_transform:
                        # modify if needed later for CoNLL
                        pass
                    else:
                        label_new = label[j]

                    label_offsets.append([label_new, s_len, len(l)])
                    # +1 to account for the whitespaces we insert below
                    s_len += len(l) + 1

                # create a single string for the sentence
                sentence = ' '.join(sentence)
                examples.append(
                    InputExample(
                        guid=guid, text=sentence, labels=label_offsets
                    )
                )
                sentence = []
                label = []
                n += 1
                guid = "%s-%s-%s" % (set_type, n, i + 1)
                continue

            if line.startswith('-DOCSTART-'):
                continue

            text = line.split(' ')
            sentence.append(text[0])
            label.append(text[3])
        return examples


class DeidProcessor(DataProcessor):
    """Processor for the de-id datasets."""
    def __init__(self, data_dir, label_transform=False):
        """Initialize a data processor with the location of the data."""
        super().__init__(data_dir, label_transform=label_transform)
        self.data_filenames = {'train': 'train', 'test': 'test'}

    def _create_examples(self, fn, set_type):  # lines, set_type):
        """Creates examples for the training, validation, and test sets."""
        examples = []

        # for de-id datasets, "fn" is a folder containing txt/ann subfolders
        # "txt" subfolder has text files with the text of the examples
        # "ann" subfolder has annotation files with the labels
        txt_path = os.path.join(fn, 'txt')
        ann_path = os.path.join(fn, 'ann')
        for f in os.listdir(txt_path):
            if not f.endswith('.txt'):
                continue

            guid = "%s-%s" % (set_type, f[:-4])
            with open(os.path.join(txt_path, f), 'r') as fp:
                text = ''.join(fp.readlines())

            # load in the annotations
            fn = os.path.join(ann_path, f'{f[:-4]}.gs')
            if self.label_transform:
                labels = label_transform(fn)
            else:
                labels = load_labels(fn)
            examples.append(InputExample(guid=guid, text=text, labels=labels))

        return examples


class i2b22006Processor(DeidProcessor):
    """Processor for deid using i2b2 2006 dataset."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_list = (
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


class i2b22014Processor(DeidProcessor):
    """Processor for deid using i2b2 2014 dataset."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_list = (
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


class DernoncourtLeeProcessor(DeidProcessor):
    """Processor for deid using Dernoncourt-Lee corpus."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_list = (
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


class PhysioNetProcessor(DeidProcessor):
    """Processor for deid using binary PHI/no phi labels."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_list = (
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


class RRProcessor(DeidProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_list = (
            # non-entities
            'O',
            # names
            'NAME',
            # professions
            # locations
            'PROTECTED_ENTITY',
            'ORGANIZATION',
            # ages
            'AGE',
            'AGE_>_89',
            # dates
            'DATE',
            # IDs
            'OTHER',
            # contacts
            'CONTACT'
        )


PROCESSORS = {
    'conll': CoNLLProcessor,
    'i2b2_2006': i2b22006Processor,
    'i2b2_2014': i2b22014Processor,
    'physionet': PhysioNetProcessor,
    'dernoncourt_lee': DernoncourtLeeProcessor,
    'rr': RRProcessor
}