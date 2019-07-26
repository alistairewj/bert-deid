import os
import pytest

from pydeid import annotation


def test_bert_deid_rr(radiology_reports, bert_i2b2_2014_model):
    for f in radiology_reports:
        text = radiology_reports[f]

        # ann with bert
        ann = bert_i2b2_2014_model.annotate(
            text, document_id=f)

        # merge intervals that are close together
        ann = annotation.merge_intervals(
            ann, dist=1, text=text)

        # post-fix to reduce false positives
        ann = bert_i2b2_2014_model.postfix(ann, text)

        assert ann.shape[0] > 0
