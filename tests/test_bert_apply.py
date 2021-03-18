"""Test applying bert to a set of text."""


def test_model_predict(bert_i2b2_2014_model, ds_text):
    truth = [
        ['NAME', 7, 8], ['NAME', 11, 4], ['NAME', 14, 1], ['NAME', 15, 1],
        ['NAME', 16, 5], ['ID', 50, 8], ['ID', 53, 5], ['ID', 55, 3],
        ['ID', 57, 1], ['DATE', 77, 2], ['DATE', 79, 1], ['DATE', 80, 2],
        ['DATE', 82, 1], ['DATE', 83, 2], ['DATE', 117, 2], ['DATE', 119, 1],
        ['DATE', 120, 2], ['DATE', 122, 1], ['DATE', 123, 2], ['DATE', 143, 2],
        ['DATE', 145, 1], ['DATE', 146, 2], ['DATE', 148, 1], ['DATE', 149, 4],
        ['NAME', 463, 8], ['NAME', 467, 4], ['NAME', 470, 1], ['AGE', 486, 3],
        ['NAME', 511, 8], ['DATE', 533, 2], ['DATE', 535, 1], ['DATE', 536, 2],
        ['DATE', 538, 1], ['DATE', 539, 2], ['DATE', 909, 2], ['DATE', 911, 1],
        ['DATE', 912, 2], ['DATE', 914, 1], ['DATE', 915, 2], ['DATE', 1422, 2],
        ['DATE', 1424, 1], ['DATE', 1425, 2], ['DATE', 1428, 2]
    ]

    labels = bert_i2b2_2014_model.predict(ds_text)

    for l, target in enumerate(truth):
        # skip probability in first element for preds
        pred = labels[l][1:]
        assert pred[0] == target[0], 'label incorrect'
        assert pred[1] == target[1], 'offset incorrect'
        assert pred[2] == target[2], 'length incorrect'


def test_model_apply(bert_i2b2_2014_model, ds_text, ds_text_deid):
    text_deid = bert_i2b2_2014_model.apply(ds_text)

    assert text_deid == ds_text_deid
