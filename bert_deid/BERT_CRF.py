import torch
from torchcrf import CRF
from torch import nn
import os
import logging

logger = logging.getLogger(__name__)
WEIGHTS_NAME= "pytorch_model.bin"


class BertCRF(nn.Module):
    def __init__(self, num_classes, bert_model, device='cpu', dropout=0.1, hidden_size=768) -> None:
        super(BertCRF, self).__init__()

        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.crf = CRF(num_tags=num_classes, batch_first=True).to(self.device)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # outputs : (last_encoder_layer, pooled_output, attention_weight)
        # return negative log-likelihood, predicted tag sequences if labels are given
        # otherwise return predicted tag sequences only
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.hidden2label(last_encoder_layer).to(self.device)

        if labels is not None:
            log_likelihood, tag_seqs = self.crf(emissions, labels), torch.Tensor(self.crf.decode(emissions)).to(self.device)
            return -1*log_likelihood, tag_seqs
        else:
            tag_seqs = torch.Tensor(self.crf.decode(emissions)).to(self.device)
            return None, tag_seqs

    def save_pretrained(self, save_dir):
        assert os.path.isdir(
            save_dir
        )
        model_to_save = self.module if hasattr(self, "module") else self
        # attach architecture to the config
        # model_to_save.config.architectures = [model_to_save.__class__.__name__]
        # # save configuration file
        # model_to_save.config.save_pretrained(save_dir)

        output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def from_pretrained(self, model_path):
        if os.path.isfile(os.path.join(model_path, WEIGHTS_NAME)):
            achieve_file = os.path.join(model_path, WEIGHTS_NAME)
        else:
            raise ValueError("No file named {} found in directory {".format(WEIGHTS_NAME, model_path))
        self.load_state_dict(torch.load(achieve_file))

