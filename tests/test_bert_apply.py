# load in a trained model


def test_model_predict(bert_i2b2_2014_model):
    text = 'Dr. Michelle Obama says I have had a pneumothorax since 2019-01-01.'
    truth = [
        ['NAME', 4, 8], ['NAME', 13, 5], ['DATE', 56, 4], ['DATE', 60, 1],
        ['DATE', 61, 2], ['DATE', 63, 1], ['DATE', 64, 2]
    ]

    preds, lengths, offsets = bert_i2b2_2014_model.predict(text)

    # get the free-text label
    labels = [
        bert_i2b2_2014_model.label_set.id_to_label[idxMax]
        for idxMax in preds.argmax(axis=1)
    ]
    outputs = [
        [labels[i], offsets[i], lengths[i]]
        for i in range(len(preds)) if labels[i] != 'O'
    ]

    for l in range(len(outputs)):
        pred = outputs[l]
        target = truth[l]
        assert pred[0] == target[0], 'label incorrect'
        assert pred[1] == target[1], 'offset incorrect'
        assert pred[2] == target[2], 'length incorrect'


def test_model_apply(bert_i2b2_2014_model):
    text = 'Hello my name is Dr. Somayah, I have worked here since 2019.'
    text_deid = 'Hello my name is Dr. ___, I have worked here since ___.'

    assert text_deid == bert_i2b2_2014_model.apply(text, repl='___')


def test_model_apply_multiple_entities(bert_i2b2_2014_model):
    text = 'Patient is a 64 yo male named Richard the Lionheart.'
    text_deid = 'Patient is a ___ yo male named ___.'

    assert text_deid == bert_i2b2_2014_model.apply(text, repl='___')
