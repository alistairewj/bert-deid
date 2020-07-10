import torch
from bert_deid.crf import CRF
from torch import nn
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"


class BertCRF(nn.Module):
    def __init__(
        self,
        num_classes,
        bert_model,
        device='cpu',
        dropout=0.1,
        hidden_size=768
    ) -> None:
        super(BertCRF, self).__init__()

        self.bert = bert_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.crf = CRF(num_tags=num_classes, batch_first=True).to(self.device)

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, labels=None
    ):
        # return negative log-likelihood, predicted tag sequences if labels are given
        # otherwise return predicted tag sequences only
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        last_encoder_layer = outputs[0]  # (batch_size, seq_length, hidden_size)
        batch_size, seq_len = last_encoder_layer.shape[:2]
        # mask all -100
        mask = (labels >= 0).long()
        # torchCRF library assume no special tokens [cls] [sep] at either front of end of sequence
        last_encoder_layer = last_encoder_layer[:, 1:-1, :]
        # update all -100 to 0 to avoid indicies out-of-bound in CRF
        labels = labels[:, 1:-1] * mask[:, 1:-1]
        mask = mask[:, 1:-1].byte()

        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.hidden2label(last_encoder_layer).to(self.device)

        if labels is not None:
            log_likelihood = self.crf(
                emissions=emissions, tags=labels, mask=mask
            )
            tag_seqs = torch.Tensor(self.crf.decode(emissions, mask=mask)).to(
                self.device
            )  # (batch_size, seq_len)
            return -1 * log_likelihood, tag_seqs

        tag_seqs = torch.Tensor(self.crf.decode(emissions,
                                                mask=mask)).to(self.device)
        return None, tag_seqs

    def save_pretrained(self, save_dir):
        """Save Bert-CRF model param weights"""
        assert os.path.isdir(save_dir)
        model_to_save = self.module if hasattr(self, "module") else self

        output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def from_pretrained(self, model_path):
        """Load Bert-CRF model param weights"""
        if os.path.isfile(os.path.join(model_path, WEIGHTS_NAME)):
            achieve_file = os.path.join(model_path, WEIGHTS_NAME)
        else:
            raise ValueError(
                "No file named {} found in directory {".format(
                    WEIGHTS_NAME, model_path
                )
            )
        self.load_state_dict(torch.load(achieve_file))
