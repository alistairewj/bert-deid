from torch import nn
import torch
import numpy as np
from bert_deid.crf import CRF
from transformers import BertModel, BertPreTrainedModel


class ModelExtraFeatureCRF(BertPreTrainedModel):
    def __init__(self, config, num_features):
        super().__init__(config)
        config.output_hidden_states = True
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.num_features = num_features
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feature_layer = nn.Linear(self.num_features, 1)

        self.hidden2label = nn.Linear(self.hidden_size + 1, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # self.classifier = nn.Linear(self.hidden_size + 1, self.num_labels)

        # self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        extra_features=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_encoder_layer = outputs[0]  # (batch_size, seq_length, hidden_size)
        batch_size, max_seq_length = last_encoder_layer.shape[:2]

        feature = self.feature_layer(
            extra_features.float().view(
                batch_size, max_seq_length, self.num_features
            )
        )
        output_layer_extra_features = torch.cat(
            [last_encoder_layer, feature], dim=-1
        )
        output_layer_extra_features = self.dropout(output_layer_extra_features)

        # mask all -100
        mask = (labels >= 0).long()
        # update all -100 to 0 to avoid indicies out-of-bound in CRF
        labels = labels * mask
        mask = mask.to(torch.uint8)  #.byte()

        emissions = self.hidden2label(output_layer_extra_features)
        best_tag_seqs = torch.Tensor(self.crf.decode(emissions, mask=mask)
                                    ).long()  # (batch_size, seq_len)
        outputs = (best_tag_seqs, )

        if labels is not None:
            log_likelihood = self.crf(
                emissions=emissions, tags=labels, mask=mask
            )
            outputs = (-1 * log_likelihood, ) + outputs
        return outputs
