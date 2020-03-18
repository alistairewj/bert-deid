import re
import csv
import os
import sys
import logging

from bert_deid.label import Label, LabelCollection

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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
    def __init__(self, data_dir):
        """Initialize a data processor with the location of the data."""
        self.data_dir = data_dir
        self.data_filenames = None
        self.label_set = None

    def get_labels(self):
        """Gets the list of labels for this data set."""
        if not hasattr(self, 'label_set'):
            raise NotImplementedError()
        else:
            return list(self.label_set.label_list)

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
    def __init__(self, data_dir, label_transform=None):
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
    def __init__(self, data_dir, label_set):
        """Initialize a data processor with the location of the data."""
        super().__init__(data_dir)
        self.data_filenames = {'train': 'train', 'test': 'test'}
        self.label_set = label_set

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

            # load the labels from a file
            # these datasets have consistent folder structures:
            #   root_path/txt/RECORD_NAME.txt - has text
            #   root_path/ann/RECORD_NAME.gs - has annotations
            self.label_set.from_csv(fn)
            examples.append(
                InputExample(
                    guid=guid, text=text, labels=self.label_set.labels
                )
            )

        return examples