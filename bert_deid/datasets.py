import os
import logging
from typing import List, Optional, Union

from filelock import FileLock
import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from bert_deid.processors import InputFeatures, TokenClassificationTask, Split

logger = logging.getLogger(__name__)


class TokenClassificationDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        token_classification_task: TokenClassificationTask,
        data_dir: str,
        tokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(max_seq_length)
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(
                    f"Loading features from cached file {cached_features_file}"
                )
                self.features = torch.load(cached_features_file)
            else:
                logger.info(
                    f"Creating features from dataset file at {data_dir}"
                )
                examples = token_classification_task.read_examples_from_file(
                    data_dir, mode
                )
                self.features = token_classification_task.convert_examples_to_features(
                    examples, labels, tokenizer
                )
                logger.info(
                    f"Saving features into cached file {cached_features_file}"
                )
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]