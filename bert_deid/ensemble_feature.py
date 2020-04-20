import pydeid
from pydeid import annotator 
import pkgutil

# load all modules on path
pkg = 'pydeid.annotator._patterns'
_PATTERN_NAMES = [name for _, name, _ in pkgutil.iter_modules(
    pydeid.annotator._patterns.__path__
)]
# _PATTERN_NAMES = PATTERN_NAMES + ['all']


def create_extra_feature_vectors(pattern_name, pattern_label, text, input_offsets, input_lengths, token_sw,
    max_seq_length=128):
    if pattern_name is None:
        return None
    if pattern_name.lower() not in _PATTERN_NAMES:
        raise ValueError("Invalid pattern argument")

    # find PHI with specific pydeid pattern
    # if pattern_name == 'all':
    #     modules = PATTERN_NAMES
    # else:
    #     modules = [pattern_name]
    modules = [pattern_name]
    model = annotator.Pattern(modules)
    pydeid_phi_df = model.annotate(text)

    # mark location of detected phi with 1s and rest with 0s
    phi_loc = [0] * len(text)
    for i, row in pydeid_phi_df.iterrows():
        start, end = row['start'], row['stop']
        phi_loc[start:end] = [pattern_label] * (end-start)


    # transform feature to match with BERT tokenization offset
    feature_vector = []
    for i in range(len(input_offsets)):
        start = input_offsets[i]
        # offset uses negative to indicate special token padding
        if start >= 0: # valid input token
            stop = start + input_lengths[i]
            if not token_sw[i]:
                # token is assigned with most occured label for correspoinding characters
                feature_vector.append(max(phi_loc[start:stop], key=list(phi_loc[start:stop]).count))
            else:
                # similarily as BERT, aggregate subword token label to the very first token
                feature_vector.append(0)

    # adds [CLS] at front
    feature_vector = [0] + feature_vector
    # padd rest with zeros
    feature_vector += [0] * (max_seq_length - len(feature_vector))

    return feature_vector

