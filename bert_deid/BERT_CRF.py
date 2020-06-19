from torch import nn
import torch
import numpy as np
from bert_deid.crf import CRF
from transformers import BertModel, BertPreTrainedModel


class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden2label = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        last_encoder_layer = outputs[0]  # (batch_size, seq_length, hidden_size)

        # mask all -100
        mask = (labels >= 0).long()
        # update all -100 to 0 to avoid indicies out-of-bound in CRF
        labels = labels * mask
        mask = mask.to(torch.uint8)  #.byte()

        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.hidden2label(last_encoder_layer)
        best_tag_seqs = torch.Tensor(self.crf.decode(emissions, mask=mask)
                                    ).long()  # (batch_size, seq_len)
        outputs = (best_tag_seqs, )

        if labels is not None:
            log_likelihood = self.crf(
                emissions=emissions, tags=labels, mask=mask
            )
            outputs = (-1 * log_likelihood, ) + outputs
        return outputs
