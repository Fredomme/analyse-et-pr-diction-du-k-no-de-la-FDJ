import numpy as np

def jaccard_similarity(set_a, set_b):
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0

def precision_at_k(pred_draw, true_draw, k=20):
    set_pred = set(pred_draw[:k])
    set_true = set(true_draw)
    return len(set_pred.intersection(set_true)) / k

def recall_at_k(pred_draw, true_draw, k=20):
    set_pred = set(pred_draw[:k])
    set_true = set(true_draw)
    return len(set_pred.intersection(set_true)) / len(set_true)

def evaluate_draws(pred_draws, true_draws):
    all_jaccard = []
    all_precision = []
    all_recall = []
    for pred, truth in zip(pred_draws, true_draws):
        set_pred = set(pred)
        set_true = set(truth)
        all_jaccard.append(jaccard_similarity(set_pred, set_true))
        all_precision.append(precision_at_k(pred, truth, k=20))
        all_recall.append(recall_at_k(pred, truth, k=20))
    return {
        'jaccard_mean': np.mean(all_jaccard),
        'precision_mean': np.mean(all_precision),
        'recall_mean': np.mean(all_recall),
    }
