import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class ModelExtraFeature(BertPreTrainedModel):
    def __init__(self, config, num_features):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.num_features = num_features
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feature_layer = nn.Linear(self.num_features, 1)
        self.classifier = nn.Linear(self.hidden_size + 1, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        extra_features=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        batch_size, max_seq_length = outputs[0].shape[:2]

        feature = self.feature_layer(
            extra_features.float().view(
                batch_size, max_seq_length, self.num_features
            )
        )
        output_layer_extra_features = torch.cat([outputs[0], feature], dim=-1)
        output_layer_extra_features = self.dropout(output_layer_extra_features)
        logits = self.classifier(output_layer_extra_features)

        outputs = (logits, )

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1),
                    torch.tensor(loss_fn.ignore_index).type_as(labels)
                )
                loss = loss_fn(active_logits, active_labels)

            else:
                loss = loss_fn(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )

            outputs = (loss, ) + outputs

        return outputs
