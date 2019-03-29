# module for evaluating performance on NER task
import os


def load_conll(fn):
    """
    Loads CoNLL formatted data.

    Args:
        fn (str): The filename containing the entities.

    Returns:
        y_true (list): The true entities.
        y_pred (list): The predicted entities.

    The assumed CoNLL format is:

    * One token per line (usually a word but can be punctuation)
    * Distinct examples (e.g. sentences) separated by a new line
    * Each line is space delimited, formatted as:
      TOKEN DOCUMENT_ID START_INDEX STOP_INDEX TRUE_LABEL PREDICTED_LABEL

    """
    # loads CoNLL format data, specifically data
    if not os.path.exists(fn):
        raise FileNotFoundError(f'Did not find CoNLL formatted file {fn}')

    with open(fn, 'r') as fp:
        # strip newline from each line as we read
        preds = [x.rstrip() for x in fp.readlines()]

    # remove empty lines
    preds = [x for x in preds if x != '']

    # split into list of lists
    preds = [x.split(' ') for x in preds]
    # get ypred
    y_true = [x[4] for x in preds]
    y_pred = [x[5] for x in preds]

    return y_true, y_pred
