"""Class for applying BERT-deid on text."""
import os
import re
import logging
from hashlib import sha256

import numpy as np
import pandas as pd
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset
)

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    BertModel,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

# custom class written for albert token classification
from bert_deid import tokenization, processors
from bert_deid.label import LABEL_SETS, LabelCollection, LABEL_MEMBERSHIP
from bert_deid.model.albert import AlbertForTokenClassification
from bert_deid.model.crf import BertCRF
from bert_deid.model.extra_feature import ModelExtraFeature

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForTokenClassification, AlbertTokenizer),
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert":
        (
            DistilBertConfig, DistilBertForTokenClassification,
            DistilBertTokenizer
        ),
    "camembert":
        (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta":
        (
            XLMRobertaConfig, XLMRobertaForTokenClassification,
            XLMRobertaTokenizer
        ),
    'bert_crf': (BertConfig, BertModel, BertTokenizer),
    'bert_extra_feature': (BertConfig, ModelExtraFeature, BertTokenizer)
}

logger = logging.getLogger(__name__)


def pool_annotations(df):
    # pool token-wise annotations together
    # this is necessary if overlapping examples are used
    # i.e. self.token_step_size < self.sequence_length
    if df.shape[0] == 0:
        return df

    # get location of maximally confident annotations
    df_keep = df.groupby(['annotator', 'start', 'stop'])[['confidence']].max()

    df_keep.reset_index(inplace=True)

    # merge on these columns to remove rows with low confidence
    grp_cols = list(df_keep.columns)
    df = df.merge(df_keep, how='inner', on=grp_cols)

    # if two rows had identical confidence, keep the first
    df.drop_duplicates(subset=grp_cols, keep='first', inplace=True)

    return df


class Transformer(object):
    """Wrapper for a Transformer model to be applied for NER."""
    def __init__(
        self,
        model_path,
        device='cpu',
        patterns=None,
    ):
        self.label_set = torch.load(os.path.join(model_path, "label_set.bin"))
        self.num_labels = len(self.label_set.label_list)
        self.patterns = patterns

        # by default, we do non-overlapping segments of text
        # self.token_step_size = token_step_size
        # sequence_length is how long each example for the model is
        # self.sequence_length = sequence_length

        # get the definition classes for the model
        training_args = torch.load(
            os.path.join(model_path, 'training_args.bin')
        )
        self.model_type = training_args.model_type

        # max seq length is what we pad the model to
        # max seq length should always be >= sequence_length + 2
        self.max_seq_length = training_args.max_seq_length

        config_class, model_class, tokenizer_class = MODEL_CLASSES[
            self.model_type]

        if self.model_type == 'bert_crf':
            # use pretrained bert model path to initialize BertCRF object
            new_model_path = training_args.model_name_or_path
        else:
            new_model_path = model_path
        # Use cross entropy ignore index as padding label id so
        # that only real label ids contribute to the loss later
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        # initialize the model
        self.config = config_class.from_pretrained(new_model_path)
        self.tokenizer = tokenizer_class.from_pretrained(new_model_path)
        model_param = {'pretrained_model_name_or_path': new_model_path}
        if self.model_type == 'bert_extra_feature':
            model_param['num_features'] = len(self.patterns)
        self.model = model_class.from_pretrained(**model_param)

        if self.model_type == 'bert_crf':
            self.model = BertCRF(
                num_classes=self.num_labels,
                bert_model=self.model,
                device=device
            )
            self.model.from_pretrained(model_path)

        # prepare the model for evaluation
        # CPU probably faster, avoids overhead
        self.device = torch.device(device)
        self.model.to(self.device)

    def split_by_overlap(self, text, token_step_size=20, sequence_length=100):
        # track offsets in tokenization
        tokens, tokens_sw, tokens_idx = self.tokenizer.tokenize_with_index(text)

        if len(tokens_idx) == 0:
            # no tokens found, return empty list
            return []
        # get start index of each token
        tokens_start = [x[0] for x in tokens_idx]
        tokens_start = np.array(tokens_start)

        # forward fill index for first token over its subsequent subword tokens
        # this means that if we try to split on a subword token, we will actually
        # split on the starting word
        mask = np.array(tokens_sw) == 1
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        tokens_start[mask] = tokens_start[idx[mask]]

        if len(tokens) <= sequence_length:
            # very short text - only create one example
            seq_offsets = [[tokens_start[0], len(text)]]
        else:
            seq_offsets = range(
                0,
                len(tokens) - sequence_length, token_step_size
            )
            last_offset = seq_offsets[-1] + token_step_size
            seq_offsets = [
                [tokens_start[x], tokens_start[x + sequence_length]]
                for x in seq_offsets
            ]

            # last example always goes to the end of the text
            seq_offsets.append([tokens_start[last_offset], len(text)])

        # turn our offsets into examples
        # create a list of lists, each sub-list has 4 elements:
        #   sentence number, start index, end index, text of the sentence
        examples = list()

        for i, (start, stop) in enumerate(seq_offsets):
            examples.append([i, start, stop, text[start:stop]])

        return examples

    def predict(self, text, batch_size=8):
        # args, model, tokenizer, processor, pad_token_label_id, mode="test"
        # sets the model to evaluation mode to fix parameters
        self.model.eval()

        # annotate a string of text
        # split the text into examples
        # we choose non-overlapping examples for evaluation
        # examples = split_by_overlap(
        #    text,
        #    self.tokenizer,
        #    token_step_size=self.token_step_size,
        #    sequence_length=self.sequence_length
        #)

        # in this case we have a length 1 example
        # we use the SHA-256 hash of the text as the globally unique identifier
        guid = sha256(text.encode()).hexdigest()
        examples = [processors.InputExample(guid=guid, text=text, labels=None)]
        features = tokenization.convert_examples_to_features(
            examples,
            self.label_set.label_to_id,
            self.max_seq_length,
            self.tokenizer,
            cls_token_at_end=bool(self.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=bool(self.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(self.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=self.tokenizer.pad_token,
            pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
            pad_token_label_id=self.pad_token_label_id,
            feature_overlap=0
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )

        if hasattr(features[0], 'additional_features'):
            all_additional_features = torch.tensor(
                [f.additional_features for f in features], dtype=torch.long
            )

            dataset = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                all_additional_features
            )
        else:
            dataset = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids
            )

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        logits = None
        out_label_ids = None
        mask = None

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3]
                }
                if self.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2]
                        if self.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                if self.model_type == 'bert_extra_feature':
                    inputs['extra_features'] = batch[4]
                outputs = self.model(**inputs)
                _, batch_logits = outputs[:2]

            # extract output predictions (logits) as a numpy array
            # N_BATCHES x N_SEQ_LENGTH x N_LABELS
            batch_logits = batch_logits.detach().cpu().numpy()
            batch_size, seq_len = batch_logits.shape[:2]
            if self.model_type == 'bert_crf':
                # BertCRF disregard first and last token,
                pad_logits = np.ones((batch_size, 1)) * self.pad_token_label_id
                batch_logits = np.concatenate(
                    (pad_logits, batch_logits, pad_logits), axis=1
                )
                # BertCRF only gives a label prediction
                # broadcast to (,,num_label) to match to BERT output
                batch_logits = np.expand_dims(batch_logits, axis=2)
                batch_logits = batch_logits.astype(int)
            if logits is None:
                logits = batch_logits
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                logits = np.append(logits, batch_logits, axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs["labels"].detach().cpu().numpy(),
                    axis=0
                )

        # re-align the predictions with the original text
        preds, offsets, lengths = [], [], []
        for f, feature in enumerate(features):
            # get predictions for only the kept labels
            idxKeep = np.where(out_label_ids[f, :] != self.pad_token_label_id
                              )[0]
            preds.append(logits[f, idxKeep, :])
            # get offsets for only the kept labels
            offsets.extend(
                [
                    x
                    for i, x in enumerate(feature.input_offsets) if i in idxKeep
                ]
            )

            # increment the lengths for the first token in words tokenized into subwords
            feature_lengths = feature.input_lengths
            for i in reversed(range(len(feature.input_subwords))):
                if feature.input_subwords[i]:
                    # cumulatively sums lengths for subwords until the first subword token
                    feature_lengths[i - 1] += feature_lengths[i]

            lengths.extend(
                [x for i, x in enumerate(feature_lengths) if i in idxKeep]
            )

        preds = np.concatenate(preds, axis=0)

        # now, we may have multiple predictions for the same offset token
        # this can happen as we are overlapping observations to maximize
        # context for tokens near the window edges
        # so we take the *last* prediction, because that prediction will have
        # the most context

        # np.unique returns index of first unique value, so reverse the list
        offsets.reverse()
        _, unique_idx = np.unique(offsets, return_index=True)
        unique_idx = len(offsets) - unique_idx - 1
        offsets.reverse()

        # align the predictions with the text offsets
        offsets = [x for i, x in enumerate(offsets) if i in unique_idx]
        lengths = [x for i, x in enumerate(lengths) if i in unique_idx]
        preds = preds[unique_idx, :]

        return preds, lengths, offsets

    def apply(self, text, repl='___'):
        preds, lengths, offsets = self.predict(text)

        # get the free-text label
        labels = [
            self.label_set.id_to_label[idxMax]
            for idxMax in preds.argmax(axis=1)
        ]

        # merge entities which are adjacent
        #removed_entities = np.zeros(len(labels), dtype=bool)
        for i in reversed(range(len(labels))):
            if i == 0 or labels[i] == 'O':
                continue

            if labels[i] == labels[i - 1]:
                offset, length = offsets.pop(i), lengths.pop(i)
                lengths[i - 1] = offset + length - offsets[i - 1]
                labels.pop(i)

        #keep_entities = ~removed_entities
        #labels = [l for i, l in enumerate(labels) if keep_entities[i]]
        #lengths = lengths[keep_entities]
        #offsets = offsets[keep_entities]
        for i in reversed(range(len(labels))):
            if labels[i] != 'O':
                # replace this instance of text with three underscores
                text = text[:offsets[i]] + repl + text[offsets[i] + lengths[i]:]

        return text
