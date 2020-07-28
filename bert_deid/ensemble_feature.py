import pydeid
import pkgutil
from pydeid.annotation import Document, EntityType
from pydeid.annotators import Pattern

from pydeid.annotators import _patterns
# load all modules on path
pkg = 'pydeid.annotators._patterns'
_PATTERN_NAMES = [
    name for _, name, _ in pkgutil.iter_modules(_patterns.__path__)
]
_PATTERN_NAMES.remove('_pattern')


def find_phi_location(pattern_name, pattern_label, text):
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

    # mark location of detected phi with 1s and rest with 0s
    phi_loc = [0] * len(text)
    for ann in txt_annotated.annotations:
        start, end = ann.start, ann.end
        phi_loc[start:end] = [pattern_label] * (end - start)

    return phi_loc


def find_phi_location_philter(df, text):
    phi_loc = [0] * len(text)
    for i, row in df.iterrows():
        start, stop = row['start'], row['stop']
        # only for date:
        if row['entity_type'].upper() == 'DATE':
            phi_loc[start:stop] = [1] * (stop - start)
    return phi_loc

def find_overlap(pydeid_loc, philter_loc):
    phi_loc = [0] * len(pydeid_loc)
    for i in range(len(pydeid_loc)):
        if pydeid_loc[i] == 1 and philter_loc[i] == 1:
            phi_loc[i] = 1
    return phi_loc 

def find_either(pydeid_loc, philter_loc):
    phi_loc = [0] * len(pydeid_loc)
    for i in range(len(pydeid_loc)):
        if pydeid_loc[i] == 1 or philter_loc[i] == 1:
            phi_loc[i] = 1
    return phi_loc 

def find_exclude(first_loc, second_loc):
    phi_loc = [0] * len(first_loc)
    for i in range(len(first_loc)):
        if first_loc[i] == 1 and second_loc[i] != 1:
            phi_loc[i] = 1
    return phi_loc


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
