# Compute psx (n x m matrix of predicted probabilities) on your own, with any classifier.
import argparse
import pickle
from pathlib import Path

import numpy as np
from cleanlab import pruning

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='/enc_data/deid-gs/i2b2_2014/test',
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--task",
        default='i2b2_2014',
        type=str,
        help="Type of dataset",
    )
    args = parser.parse_args()
    label_map = {
        0: 'O',
        1: 'DOCTOR',
        2: 'PATIENT',
        3: 'USERNAME',
        4: 'PROFESSION',
        5: 'LOCATION',
        6: 'HOSPITAL',
        7: 'ORGANIZATION',
        8: 'URL',
        9: 'STREET',
        10: 'STATE',
        11: 'CITY',
        12: 'COUNTRY',
        13: 'ZIP',
        14: 'LOCATION-OTHER',
        15: 'AGE',
        16: 'DATE',
        17: 'BIOID',
        18: 'DEVICE',
        19: 'HEALTHPLAN',
        20: 'IDNUM',
        21: 'MEDICALRECORD',
        22: 'EMAIL',
        23: 'FAX',
        24: 'PHONE'
    }

    with open('preds.pkl', 'rb') as fp:
        files, preds, labels, lengths, offsets = pickle.load(fp)

    # align the files/lengths/offsets arrays with the labels one

    # files has 1 string per file so duplicate it for each token
    files = [[f] * len(offsets[i]) for i, f in enumerate(files)]
    # flatten the lists
    files = [f for ff in files for f in ff]
    offsets = np.asarray([o for oo in offsets for o in oo])
    lengths = np.asarray([l for ll in lengths for l in ll])

    labels = np.asarray(labels)
    #y = np.zeros([labels.shape[0], preds.shape[1]])
    #y[range(len(labels)), labels] = 1

    # convert preds from logit to probs
    preds = 1 / (1 + np.exp(-1 * preds))

    # drop a category if it has less than 5 examples
    labels_to_remove = []
    for j in range(np.max(labels)):
        if np.sum(labels == j) < 5:
            labels_to_remove.append(j)
    labels_to_remove = set(labels_to_remove)

    idxKeep = [
        i for i in range(len(labels)) if labels[i] not in labels_to_remove
    ]
    labels = labels[idxKeep]
    preds = preds[idxKeep, :]
    offsets = offsets[idxKeep]
    lengths = lengths[idxKeep]
    files = [files[i] for i in idxKeep]

    preds = preds[:, [
        j for j in range(preds.shape[1]) if j not in labels_to_remove
    ]]

    # re-map the label IDs so they are sequential integers
    label_id_map = {l: i for i, l in enumerate(np.unique(labels))}
    labels = [label_id_map[l] for l in labels]
    labels = np.array(labels)

    label_map = {
        label_id_map[l]: v
        for l, v in label_map.items() if l in label_id_map
    }

    # Be sure you compute probs in a holdout/out-of-sample manner (e.g. cross-validation)
    # Now getting label errors is trivial with cleanlab... its one line of code.
    # Label errors are ordered by likelihood of being an error. First index is most likely error.

    if preds.shape[0] > 100000:
        print('Large predictions take a long time. Only using top 100,000.')
        preds = preds[:100000, :]
        labels = labels[:100000]

    ordered_label_errors = pruning.get_noise_indices(
        s=labels,
        psx=preds,
        sorted_index_method='normalized_margin',  # Orders label errors
    )

    data_path = Path(args.data_dir)
    text_path = data_path / 'txt'

    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    n = 0
    n_skipped = 0
    for i in ordered_label_errors:
        start, stop = offsets[i], offsets[i] + lengths[i]

        y = label_map[labels[i]]
        yhat = label_map[np.argmax(preds[i])]

        with open(text_path / files[i], 'r') as fp:
            text = ''.join(fp.readlines())

        # not interested in printing the punctuation ones
        if (stop - start) == 1:
            if text[start:stop] in ('.', ',', ':', ')', '()'):
                n_skipped += 1
                continue

        print(f'{files[i]}. Given: {y}. Guess: {yhat}.', end=' ')
        text = text.replace('\n', ' ')
        print(text[max(start - 30, 0):start], end='')
        print(bcolors.FAIL, end='')
        print(text[start:stop], end='')
        print(bcolors.ENDC, end='')
        print(text[stop:min(stop + 40, len(text))], end='')
        print()
        n += 1

        if n > 100:
            break

    print(f'===\nSkipped {n_skipped} punctuation mismatches.')
