import re
import csv
import os
import sys
import logging

from bert_deid.utils import create_hamonize_label_dict

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
            self._read_csv(os.path.join(data_dir, "test.csv")), "dev")

    def get_labels(self):
        # created by subclass!
        # outputs the list of valid labels
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            # dataset has stored labels as a list of offsets in 3rd column
            # e.g.
            #   [[36, 46, 'DATE'], [60, 62, 'AGE'], [165, 176, 'DATE']]
            # call eval on the string to convert this to a list
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


class binaryDeidProcessor(DeidProcessor):
    """Processor for deid using binary PHI/no phi labels."""

    def get_labels(self):
        return [
            'PHI', 'O'
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


class RadReportLabelProcessor(DataProcessor):
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
            self._read_csv(os.path.join(data_dir, "test.csv")), "dev")

    def get_labels(self):
        return [
            'CABG', 'COPD_signs', 'Dobbhoff_tube', 'LAP', 'LV_predominance',
            'NG_tube', 'PICC', 'Port-A', 'Swan-Ganz', 'abnormal_foreign_body',
            'abscess', 'aeration', 'air_bronchogram', 'air_content', 'air_fluid_level',
            'air_space_pattern', 'air_trapping', 'alveolar_edema', 'alveolar_pattern', 'aortic_aneurysm',
            'aortic_calcification', 'aortic_dilatation', 'aortic_elongation', 'artificial_heart_valve', 'asbestosis_signs',
            'ascites', 'aspiration', 'atelectasis', 'azygous_distension', 'azygous_fissure',
            'bronchiectasis', 'bullas__blebs', 'calcified_granuloma', 'calcified_node', 'calcified_pleural_plaque',
            'callus', 'cardiac_silhouette_enlargement', 'cardiomediastinal_enlargement', 'cardiomegaly', 'cardiomyopathy', 'cavitation', 'central_venous_catheter', 'cephalization', 'cervical_rib', 'chest_drain_tube', 'chest_tube', 'chronic_changes', 'chronic_lung_disease', 'clear_lung', 'clip', 'coiling_tube', 'collapse', 'congestive_heart_failure', 'consolidation', 'correlation', 'cp_angle_blunting', 'crowding_bronchovascular_markings', 'cyst', 'cystic_lesion', 'deformity', 'demineralization', 'density', 'displaced_fracture', 'displaced_fracture_clavicle', 'displaced_fracture_rib', 'ekg_lead', 'emphysema', 'empyema', 'endotracheal_tube', 'exclude', 'fissural_effusion', 'fissure_thickening', 'fixation_plate__screws', 'flattened_diaphragm', 'fluid_collection', 'fluid_overload', 'follow_up', 'fracture', 'fracture_clavicle', 'fracture_humerus', 'fracture_rib', 'goiter', 'granuloma', 'granulomatous_disease', 'ground_glass_pattern', 'haziness', 'healed_fracture', 'healed_fracture_clavicle', 'healed_fracture_humerus', 'healed_fracture_rib', 'heart_valve_calcified', 'hemidiaphragm_elevation', 'hiatal_hernia', 'hilar_congestion', 'hilar_enlargement', 'hydropneumothorax', 'hyperinflated', 'hypoexpansion', 'increase_AP_diameter', 'increase_density', 'increase_marking', 'increase_opacity', 'increased_retrosternal_space', 'infection', 'infiltrates', 'interstial_edema', 'interstitial_edema', 'interstitial_lung_disease', 'interstitial_pattern', 'kerley_lines', 'kyphosis', 'linear_atelectasis', 'lobectomy', 'loculated_effusion__pneumothorax', 'lucency', 'lymph_node', 'lytic_bone_lesion', 'mass', 'mediastinal_enlargement', 'metallic', 'metastasis', 'midline_shift', 'monitoring_and_support_device', 'multifocal_pneumonia', 'nipple_shadow', 'no_bone_anomaly', 'no_cardiopul_process', 'no_complication', 'no_pleural_abnormality', 'no_pneumoperitoneum', 'nodular_pattern', 'nodule', 'normal', 'normal_abdomen', 'obscured_border', 'old_fracture', 'old_fracture_clavicle', 'old_fracture_rib', 'old_tuberculosis', 'opacity', 'osteopenia', 'osteoporosis', 'pacemaker', 'paratracheal_stripe', 'partial_collapse', 'peribronchial_thickening', 'pericardial_effusion', 'pericardial_thickening', 'perihilar_fullness', 'pleural_effusion', 'pleural_plaques', 'pleural_thickening', 'pneumatocele', 'pneumomediastinum', 'pneumonia', 'pneumothorax', 'post_radiotherapy_changes', 'postop_change', 'prosthesis', 'pulmonary_edema', 'pulmonary_fibrosis', 'pulmonary_hemorrhage', 'pulmonary_hypertension', 'pulmonary_venous_congestion', 'removed', 'resection__excision', 'respiratory_distress_syndrome', 'reticular_pattern', 'reticulonodular_pattern', 'right_sided_aortic_arch', 'round_atelectasis', 'round_density', 'sarcoidosis', 'scarring', 'sclerotic_bone_lesion', 'scoliosis', 'segmental_atelectasis', 'septal_thickening', 'shallow_inspiration', 'soft_tissue_density', 'standard_position', 'sternotomy', 'subcutaneous_emphysema', 'suboptimal_study', 'superior_mediastinal_enlargement', 'supine_position', 'suture_material', 'tension_pneumothorax', 'thyroid_enlargement', 'tortuous_aorta', 'total_collapse', 'tracheostomy', 'tuberculosis', 'tumor__malignancy', 'unchanged', 'unfolding_aorta', 'vascular_congestion', 'vascular_indistinctness', 'vascular_plethora', 'vascular_redistribution', 'vertebral_compression', 'vertebral_degenerative_changes', 'vertebral_fracture', 'vertebral_wedge_deformity', 'volume_loss', 'well_expanded_lung', 'wire',
            'no_finding'
        ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            label = line[2:]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class RadReportLabelPreprocessor(object):
    def __init__(self, min_len=2, stopwords_path=None):
        self.min_len = min_len
        self.stopwords_path = stopwords_path
        self.replacement = {
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "I would",
            "i'll": "I will",
            "i'm": "I am",
            "isn't": "is not",
            "it's": "it is",
            "it'll": "it will",
            "i've": "I have",
            "let's": "let us",
            "mightn't": "might not",
            "mustn't": "must not",
            "shan't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "weren't": "were not",
            "we've": "we have",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where's": "where is",
            "who'd": "who would",
            "who'll": "who will",
            "who're": "who are",
            "who's": "who is",
            "who've": "who have",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
            "'re": " are",
            "wasn't": "was not",
            "we'll": " will",
            "tryin'": "trying",
        }
        self.reset()

    def lower(self, sentence):
        return sentence.lower()

    def reset(self):
        if self.stopwords_path:
            with open(self.stopwords_path, 'r') as fr:
                self.stopwords = {}
                for line in fr:
                    word = line.strip(' ').strip('\n')
                    self.stopwords[word] = 1

    # delete sentence with length less than min_len
    def clean_length(self, sentence):
        if len([x for x in sentence]) >= self.min_len:
            return sentence

    def replace(self, sentence):
        # Replace words like gooood to good
        sentence = re.sub(r'(\w)\1{2,}', r'\1\1', sentence)
        # Normalize common abbreviations
        words = sentence.split(' ')
        words = [self.replacement[word]
                 if word in self.replacement else word for word in words]
        sentence_repl = " ".join(words)
        return sentence_repl

    def remove_website(self, sentence):
        sentence_repl = sentence.replace(r"http\S+", "")
        sentence_repl = sentence_repl.replace(r"https\S+", "")
        sentence_repl = sentence_repl.replace(r"http", "")
        sentence_repl = sentence_repl.replace(r"https", "")
        return sentence_repl

    def remove_name_tag(self, sentence):
        # Remove name tag
        sentence_repl = sentence.replace(r"@\S+", "")
        return sentence_repl

    def remove_time(self, sentence):
        # Remove time related text
        sentence_repl = sentence.replace(
            r'\w{3}[+-][0-9]{1,2}\:[0-9]{2}\b', "")  # e.g. UTC+09:00
        sentence_repl = sentence_repl.replace(
            r'\d{1,2}\:\d{2}\:\d{2}', "")  # e.g. 18:09:01
        sentence_repl = sentence_repl.replace(
            r'\d{1,2}\:\d{2}', "")  # e.g. 18:09
        # Remove date related text
        # e.g. 11/12/19, 11-1-19, 1.12.19, 11/12/2019
        sentence_repl = sentence_repl.replace(
            r'\d{1,2}(?:\/|\-|\.)\d{1,2}(?:\/|\-|\.)\d{2,4}', "")
        # e.g. 11 dec, 2019   11 dec 2019   dec 11, 2019
        sentence_repl = sentence_repl.replace(
            r"([\d]{1,2}\s(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s[\d]{1,2})(\s|\,|\,\s|\s\,)[\d]{2,4}",
            "")
        # e.g. 11 december, 2019   11 december 2019   december 11, 2019
        sentence_repl = sentence_repl.replace(
            r"[\d]{1,2}\s(january|february|march|april|may|june|july|august|september|october|november|december)(\s|\,|\,\s|\s\,)[\d]{2,4}",
            "")
        return sentence_repl

    def remove_breaks(self, sentence):
        # Remove line breaks
        sentence_repl = sentence.replace("\r", "")
        sentence_repl = sentence_repl.replace("\n", "")
        sentence_repl = re.sub(r"\\n\n", ".", sentence_repl)
        return sentence_repl

    def remove_ip(self, sentence):
        # Remove phone number and IP address
        sentence_repl = sentence.replace(r'\d{8,}', "")
        sentence_repl = sentence_repl.replace(
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', "")
        return sentence_repl

    def adjust_common(self, sentence):
        # Adjust common abbreviation
        sentence_repl = sentence.replace(r" you re ", " you are ")
        sentence_repl = sentence_repl.replace(r" we re ", " we are ")
        sentence_repl = sentence_repl.replace(r" they re ", " they are ")
        sentence_repl = sentence_repl.replace(r"@", "at")
        return sentence_repl

    def full2half(self, sentence):
        ret_str = ''
        for i in sentence:
            if ord(i) >= 33 + 65248 and ord(i) <= 126 + 65248:
                ret_str += chr(ord(i) - 65248)
            else:
                ret_str += i
        return ret_str

    def remove_stopword(self, sentence):
        words = sentence.split()
        x = [word for word in words if word not in self.stopwords]
        return " ".join(x)

    # main
    def __call__(self, sentence):
        x = sentence
        x = self.lower(x)
        x = self.replace(x)
        x = self.remove_website(x)
        x = self.remove_name_tag(x)
        x = self.remove_time(x)
        x = self.remove_breaks(x)
        x = self.remove_ip(x)
        x = self.adjust_common(x)
        return x
