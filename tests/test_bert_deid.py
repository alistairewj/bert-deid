import os
import pytest


def test_output_of_pretrained_bert_on_rad_reports(
    radiology_reports, bert_i2b2_2014_model
):
    """
    Test to ensure that the pre-trained model returns annotations for the
    sample radiology reports.
    """
    cols = [
        'document_id', 'annotation_id', 'annotator', 'start', 'stop', 'entity',
        'entity_type', 'comment', 'confidence'
    ]

    # if the model is not run in eval() mode, output will differ on each run
    expected_rows = [69, 39, 45, 66, 51, 49]

    for f, n_rows in zip(radiology_reports, expected_rows):
        text = radiology_reports[f]

        # ann with bert
        ann = bert_i2b2_2014_model.annotate(text, document_id=f)

        # are the column headers as expected?
        assert (cols == ann.columns).all()

        # are there annotations?
        assert ann.shape[0] > 0

        # are there the expected number of annotations?
        assert len(ann) == n_rows


def test_output_of_pretrained_bert_on_rad_reports_with_merge(
    radiology_reports, bert_i2b2_2014_model
):
    """
    Test to ensure that the pre-trained model returns annotations for the
    sample radiology reports after intervals are merged.
    """
    cols = [
        'document_id', 'annotation_id', 'annotator', 'start', 'stop', 'entity',
        'entity_type', 'comment', 'confidence'
    ]

    # if the model is not run in eval() mode, output will differ on each run
    expected_rows = [22, 14, 13, 19, 18, 16]

    for f, n_rows in zip(radiology_reports, expected_rows):
        text = radiology_reports[f]

        # ann with bert
        ann = bert_i2b2_2014_model.annotate(text, document_id=f)

        # merge intervals that are close together
        # ann = annotation.merge_intervals(ann, dist=1, text=text)

        # post-fix to reduce false positives
        # ann = bert_i2b2_2014_model.postfix(ann, text)

        # are the column headers as expected?
        assert (cols == ann.columns).all()

        # are there annotations?
        assert ann.shape[0] > 0

        # are there the expected number of annotations?
        assert len(ann) == n_rows