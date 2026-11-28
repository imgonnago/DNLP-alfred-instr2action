def evaluate(predict_fn, data, unk_word_id):
    step_correct = 0
    step_total=0
    oov_correct = 0
    oov_total = 0
    seq_correct = 0
    predictions = []

    for gold, steps in data:
        pred = predict_fn(steps)
        predictions.append(pred)
        seq_correct += (pred == gold)
        for p, g, sent in zip(pred, gold, steps):
            step_correct += (p == g)
            step_total += 1
            if unk_word_id in sent:
                oov_correct += (p == g)
                oov_total += 1

    return{
        "step_acc": step_correct / step_total,
        "seq_exact": seq_correct / len(data),
        "oov_sent_acc": oov_correct / oov_total if oov_total else None,
        "oov_sent_n": oov_total,
    }, predictions
