def precision(y_true, y_pred, average='micro'):
    true_entities = set([item for sublist in y_true for item in sublist])
    pred_entities = set([item for sublist in y_pred for item in sublist])

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall(y_true, y_pred, average='micro'):
    true_entities = set([item for sublist in y_true for item in sublist])
    pred_entities = set([item for sublist in y_pred for item in sublist])

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def f1(y_true, y_pred, average='micro'):
    true_entities = set([item for sublist in y_true for item in sublist])
    pred_entities = set([item for sublist in y_pred for item in sublist])

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score
