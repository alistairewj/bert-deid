import csv
import os
import sys
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file, delimiter=',', quotechar='"'):
        """Reads a comma separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return cls._read_file(input_file, delimiter='\t', quotechar=quotechar)

    @classmethod
    def _read_csv(cls, input_file, quotechar='"'):
        """Reads a comma separated value file."""
        return cls._read_file(input_file, delimiter=',', quotechar=quotechar)


class DeidProcessor(DataProcessor):
    """Processor for the de-id datasets with harmonized labels."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        # created by subclass!
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            # dataset has stored labels as a list of offsets
            # trust the data and call eval on the string
            label = eval(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class i2b2DeidProcessor(DeidProcessor):
    """Processor for deid using i2b2 labels."""

    def get_labels(self):
        return [
            'NAME', 'LOCATION', 'AGE',
            'DATE', 'ID', 'CONTACT', 'O',
            'PROFESSION'
        ]


class hipaaDeidProcessor(DeidProcessor):
    """Processor for deid using HIPAA labels."""

    def get_labels(self):
        return [
            'NAME', 'LOCATION', 'AGE',
            'DATE', 'ID', 'CONTACT', 'O'
        ]


class CoNLLProcessor(DataProcessor):
    """Processor for the gold standard de-id data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['[CLS]', '[SEP]', 'O',
                'B-LOC', 'B-MISC', 'B-ORG', 'B-PER',
                'I-LOC', 'I-MISC', 'I-ORG', 'I-PER']
        # return ['[CLS]', '[SEP]', 'LOC', 'MISC', 'ORG', 'PER', 'O']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentence = []
        label = []
        # n keeps track of the number of samples processed
        n = 0
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
                    label_offsets.append(
                        [s_len, s_len+len(l), label_new])
                    # +1 to account for the whitespaces we insert below
                    s_len += len(l) + 1

                # create a single string for the sentence
                sentence = ' '.join(sentence)
                examples.append(
                    InputExample(guid=guid,
                                 text_a=sentence,
                                 label=label_offsets))
                sentence = []
                label = []
                n += 1
                guid = "%s-%s-%s" % (set_type, n, i+1)
                continue

            line = line[0]
            if line.startswith('-DOCSTART-'):
                continue

            text = line.split(' ')
            sentence.append(text[0])
            label.append(text[3])
        return examples
