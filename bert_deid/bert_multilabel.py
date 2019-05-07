import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class BertFinalPooler(nn.Module):
    def __init__(self, hidden_size, n=1):
        super(BertFinalPooler, self).__init__()
        self.dense = nn.Linear(hidden_size*n, hidden_size*n)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertMultiLabel(BertPreTrainedModel):
    def __init__(self, config, num_labels, output_features='Concat_Last_Four'):
        super(BertMultiLabel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        n = 1
        if output_features == 'Finetune_All':
            n = config.num_hidden_layers
        elif output_features == 'Second_to_Last':
            n = config.num_hidden_layers-1
        elif output_features == 'Concat_Last_Four':
            n = 4
        self.output_features = output_features

        self.pooler = BertFinalPooler(config.hidden_size, n)

        self.classifier = nn.Linear(
            in_features=config.hidden_size*n,
            out_features=num_labels
        )
        self.apply(self.init_bert_weights)
        self.unfreeze_bert_encoder()

    def freeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def get_output_features(self, encoded_layers):
        """
        Concatenate one or more hidden layers for final linear layer.
        """
        if self.output_features == 'Finetune_All':
            sequence_output = torch.cat(encoded_layers, 2)
        elif self.output_features == 'First':
            sequence_output = encoded_layers[0]
        elif self.output_features == 'Second_to_Last':
            sequence_output = torch.cat(encoded_layers[1:], 1)
        elif self.output_features == 'Sum_Last_Four':
            sequence_output = sum(encoded_layers[-4:])
        elif self.output_features == 'Concat_Last_Four':
            sequence_output = torch.cat(encoded_layers[-4:], 2)
        elif self.output_features == 'Sum_All':
            sequence_output = sum(encoded_layers)
        else:
            sequence_output = encoded_layers

        pooled_output = self.pooler(sequence_output)
        return pooled_output

    def forward(self, input_ids, token_type_ids, attention_mask,
                label_ids=None,
                output_all_encoded_layers=True):
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=output_all_encoded_layers
        )

        if self.output_features != 'Last':
            pooled_output = self.get_output_features(encoded_layers)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
