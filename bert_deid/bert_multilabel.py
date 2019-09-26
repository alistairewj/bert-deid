
import os

import pandas as pd
import torch
import torch.nn as nn

from pytorch_transformers.modeling_bert import (BertConfig, WEIGHTS_NAME,
                                                CONFIG_NAME,
                                                BertPreTrainedModel, BertModel)

from bert_deid.create_csv import split_by_overlap, split_by_sentence, split_report_into_sections
from bert_deid.tokenization import BertTokenizerNER
from bert_deid.processors import RadReportLabelPreprocessor


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


class BertForRadReportLabel(BertMultiLabel):
    """BERT model for deidentification."""

    def __init__(self, model_dir):
        self.labels = [
            'CABG', 'COPD_signs', 'Dobbhoff_tube', 'LAP', 'LV_predominance',
            'NG_tube', 'PICC', 'Port-A', 'Swan-Ganz', 'abnormal_foreign_body',
            'abscess', 'aeration', 'air_bronchogram', 'air_content', 'air_fluid_level',
            'air_space_pattern', 'air_trapping', 'alveolar_edema', 'alveolar_pattern', 'aortic_aneurysm',
            'aortic_calcification', 'aortic_dilatation', 'aortic_elongation', 'artificial_heart_valve', 'asbestosis_signs',
            'ascites', 'aspiration', 'atelectasis', 'azygous_distension', 'azygous_fissure',
            'bronchiectasis', 'bullas__blebs', 'calcified_granuloma', 'calcified_node', 'calcified_pleural_plaque',
            'callus', 'cardiac_silhouette_enlargement', 'cardiomediastinal_enlargement', 'cardiomegaly', 'cardiomyopathy',
            'cavitation', 'central_venous_catheter', 'cephalization', 'cervical_rib', 'chest_drain_tube',
            'chest_tube', 'chronic_changes', 'chronic_lung_disease', 'clear_lung', 'clip',
            'coiling_tube', 'collapse', 'congestive_heart_failure', 'consolidation', 'correlation',
            'cp_angle_blunting', 'crowding_bronchovascular_markings', 'cyst', 'cystic_lesion', 'deformity',
            'demineralization', 'density', 'displaced_fracture', 'displaced_fracture_clavicle', 'displaced_fracture_rib',
            'ekg_lead', 'emphysema', 'empyema', 'endotracheal_tube', 'exclude',
            'fissural_effusion', 'fissure_thickening', 'fixation_plate__screws', 'flattened_diaphragm', 'fluid_collection',
            'fluid_overload', 'follow_up', 'fracture', 'fracture_clavicle', 'fracture_humerus',
            'fracture_rib', 'goiter', 'granuloma', 'granulomatous_disease', 'ground_glass_pattern',
            'haziness', 'healed_fracture', 'healed_fracture_clavicle', 'healed_fracture_humerus', 'healed_fracture_rib',
            'heart_valve_calcified', 'hemidiaphragm_elevation', 'hiatal_hernia', 'hilar_congestion', 'hilar_enlargement',
            'hydropneumothorax', 'hyperinflated', 'hypoexpansion', 'increase_AP_diameter', 'increase_density',
            'increase_marking', 'increase_opacity', 'increased_retrosternal_space', 'infection', 'infiltrates',
            'interstial_edema', 'interstitial_edema', 'interstitial_lung_disease', 'interstitial_pattern', 'kerley_lines',
            'kyphosis', 'linear_atelectasis', 'lobectomy', 'loculated_effusion__pneumothorax', 'lucency',
            'lymph_node', 'lytic_bone_lesion', 'mass', 'mediastinal_enlargement', 'metallic',
            'metastasis', 'midline_shift', 'monitoring_and_support_device', 'multifocal_pneumonia', 'nipple_shadow',
            'no_bone_anomaly', 'no_cardiopul_process', 'no_complication', 'no_pleural_abnormality', 'no_pneumoperitoneum',
            'nodular_pattern', 'nodule', 'normal', 'normal_abdomen', 'obscured_border',
            'old_fracture', 'old_fracture_clavicle', 'old_fracture_rib', 'old_tuberculosis', 'opacity',
            'osteopenia', 'osteoporosis', 'pacemaker', 'paratracheal_stripe', 'partial_collapse',
            'peribronchial_thickening', 'pericardial_effusion', 'pericardial_thickening', 'perihilar_fullness', 'pleural_effusion',
            'pleural_plaques', 'pleural_thickening', 'pneumatocele', 'pneumomediastinum', 'pneumonia',
            'pneumothorax', 'post_radiotherapy_changes', 'postop_change', 'prosthesis', 'pulmonary_edema',
            'pulmonary_fibrosis', 'pulmonary_hemorrhage', 'pulmonary_hypertension', 'pulmonary_venous_congestion', 'removed',
            'resection__excision', 'respiratory_distress_syndrome', 'reticular_pattern', 'reticulonodular_pattern', 'right_sided_aortic_arch',
            'round_atelectasis', 'round_density', 'sarcoidosis', 'scarring', 'sclerotic_bone_lesion',
            'scoliosis', 'segmental_atelectasis', 'septal_thickening', 'shallow_inspiration', 'soft_tissue_density',
            'standard_position', 'sternotomy', 'subcutaneous_emphysema', 'suboptimal_study', 'superior_mediastinal_enlargement',
            'supine_position', 'suture_material', 'tension_pneumothorax', 'thyroid_enlargement', 'tortuous_aorta',
            'total_collapse', 'tracheostomy', 'tuberculosis', 'tumor__malignancy', 'unchanged',
            'unfolding_aorta', 'vascular_congestion', 'vascular_indistinctness', 'vascular_plethora', 'vascular_redistribution',
            'vertebral_compression', 'vertebral_degenerative_changes', 'vertebral_fracture', 'vertebral_wedge_deformity', 'volume_loss',
            'well_expanded_lung', 'wire', 'no_finding'
        ]
        self.num_labels = len(self.labels)
        self.label_id_map = {i: label for i, label in enumerate(self.labels)}
        self.bert_model = 'bert-base-uncased'
        self.do_lower_case = True
        self.preprocessor = RadReportLabelPreprocessor()

        # by default, we do non-overlapping segments of text
        self.max_seq_length = 100
        self.token_step_size = 100

        # load trained config/weights
        model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config_file = os.path.join(model_dir, CONFIG_NAME)
        if os.path.exists(model_file) & os.path.exists(config_file):
            print(f'Loading model and configuration from {model_dir}.')
            config = BertConfig.from_json_file(config_file)
            super(BertForRadReportLabel, self).__init__(
                config, self.num_labels)
            self.load_state_dict(torch.load(model_file, map_location="cpu"))
        else:
            raise ValueError('Folder %s did not have model and config file.',
                             model_dir)

        # initialize tokenizer
        self.tokenizer = BertTokenizerNER.from_pretrained(
            self.bert_model, do_lower_case=self.do_lower_case)

        # prepare the model for evaluation
        # CPU probably faster, avoids overhead
        device = torch.device("cpu")
        self.to(device)

    def _prepare_tokens(self, tokens, tokens_sw, tokens_idx):
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return input_ids, input_mask, segment_ids

    def annotate(self, text, annotations=None, document_id=None, column=None,
                 **kwargs):
        # annotate a string of text

        # split text into sections
        sections, section_names, section_idx = split_report_into_sections(text)

        # isolate to only a few sections
        examples = []
        for s, sname in enumerate(section_names):
            if sname.lower() in ('impression', 'findings'):
                s_txt = sections[s]

                # preprocess the text
                # lower cases, splits/fixes some contractions
                # removes times/ip addresses/websites
                s_txt = self.preprocessor(s_txt)

                # split the text into list of lists, each sub-list has:
                #   sent number, start index, end index, sentence text
                # we choose non-overlapping examples for evaluation
                s_examples = split_by_sentence(s_txt)
                for ex in s_examples:
                    examples.append([
                        s*100 + ex[0],
                        ex[1] + section_idx[s],
                        ex[2] + section_idx[s],
                        ex[3]
                    ])

        preds = list()
        scores = list()
        offsets = list()
        sentences = list()

        # examples is a list of lists, each sub-list has 4 elements:
        #   sentence number, start index, end index, text of the sentence
        for e, example in enumerate(examples):
            # track offsets in tokenization
            tokens, tokens_sw, tokens_idx = self.tokenizer.tokenize_with_index(
                example[3])

            # offset index of predictions based upon example start
            tokens_offset = example[1]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens_sw = [0] + tokens_sw + [0]
            tokens_idx = [[-1]] + tokens_idx + [[-1]]

            input_ids, input_mask, segment_ids = self._prepare_tokens(
                tokens, tokens_sw, tokens_idx
            )

            # convert to tensor
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)

            # reshape to [1, SEQ_LEN] as this is expected by model
            input_ids = input_ids.view([1, input_ids.size(0)])
            input_mask = input_mask.view([1, input_mask.size(0)])
            segment_ids = segment_ids.view([1, segment_ids.size(0)])

            # make predictions
            # logits is a numpy array sized: 1 x 208
            with torch.no_grad():
                logits = self.forward(input_ids, segment_ids, input_mask)

            # remove [CLS] at beginning
            tokens = tokens[1:]
            tokens_sw = tokens_sw[1:]
            tokens_idx = tokens_idx[1:]

            # subselect pred and mask down to length of tokens
            last_token = tokens.index('[SEP]')
            tokens = tokens[:last_token]
            tokens_sw = tokens_sw[:last_token]
            tokens_idx = tokens_idx[:last_token]

            # no data for predictions - skip writing out
            if len(tokens) == 0:
                continue

            logits = logits.sigmoid()

            # decide which labels are true using a threshold
            th = 0.5
            pred = (logits > th).tolist()

            # convert to list
            scores.append(logits[0].tolist())
            offsets.append(tokens_offset)
            preds.append(pred)
            sentences.append(example[3])

        df_score = pd.DataFrame(scores, columns=self.labels)
        # retain the starting offset of the sentence examined
        df_score['offset'] = offsets
        df_score['text'] = sentences
        # df_pred = pd.DataFrame(preds, columns=self.labels)

        return df_score
