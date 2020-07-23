import pydeid
import pkgutil
from pydeid.annotation import Document, EntityType
from pydeid.annotators import Pattern
from pydeid.annotators import _patterns

from bert_deid.processors import Tag

# load all modules on path
pkg = 'pydeid.annotators._patterns'
_PATTERN_NAMES = [
    name for _, name, _ in pkgutil.iter_modules(_patterns.__path__)
]
_PATTERN_NAMES.remove('_pattern')


def apply_pydeid_to_text(text, patterns=_PATTERN_NAMES):
    additional_features = []
    for pattern_name in _PATTERN_NAMES:
        if pattern_name == 'all':
            continue
        if pattern_name is None:
            return None
        if pattern_name.lower() not in _PATTERN_NAMES:
            raise ValueError("Invalid pattern argument")

        doc = Document(text)

        # find PHI with specific pydeid pattern
        entity_types = [EntityType(pattern_name)]
        modules = [pattern_name]
        model = Pattern(modules, entity_types)
        txt_annotated = model.annotate(doc)

        # retain the detected instances
        for ann in txt_annotated.annotations:
            start, end = ann.start, ann.end
            # Tag is a named tuple with three attributes: name, start, offset
            additional_features.append(Tag(pattern_name, start, end - start))

    if 'all' in pattern_name:
        # find PHI with all pydeid pattern
        entity_types = [EntityType(name) for name in _PATTERN_NAMES]
        model = Pattern(_PATTERN_NAMES, entity_types)
        txt_annotated = model.annotate(doc)

        # TODO: merge intervals
        for ann in txt_annotated.annotations:
            start, end = ann.start, ann.end
            additional_features.append(Tag('all', start, end - start))

    return additional_features


def create_extra_feature_vector(
    phi_loc, input_offsets, input_lengths, token_sw, max_seq_length=128
):
    # transform feature to match with BERT tokenization offset
    feature_vector = []
    for i in range(len(input_offsets)):
        start = input_offsets[i]
        # offset uses negative to indicate special token padding
        if start >= 0:  # valid input token
            stop = start + input_lengths[i]
            if not token_sw[i]:
                # token is assigned with most occured label for correspoinding characters
                feature_vector.append(
                    max(
                        phi_loc[start:stop],
                        key=list(phi_loc[start:stop]).count
                    )
                )
            else:
                # similarily as BERT, aggregate subword token label to the very first token
                feature_vector.append(0)

    # adds [CLS] at front
    feature_vector = [0] + feature_vector
    # padd rest with zeros
    feature_vector += [0] * (max_seq_length - len(feature_vector))

    return feature_vector
