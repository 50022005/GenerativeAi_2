import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .data import LABELS, ALLOWED

def evidence_metrics(rows):
    counts = np.zeros(3, dtype=float)
    per = []
    empty = 0
    for row in rows:
        gold, pred = set(row['gold_evidence']), set(row['prediction']['evidence'])
        tp, fp, fn = len(gold & pred), len(pred-gold), len(gold-pred)
        counts += [tp, fp, fn]
        if not gold and not pred:
            p = r = f = 1.
            empty += 1
        else:
            p, r = tp/max(1, tp+fp), tp/max(1, tp+fn)
            f = 2*tp/max(1, 2*tp+fp+fn)
        per.append([p, r, f])
    tp, fp, fn = counts
    return {'micro_precision': tp/max(1, tp+fp), 'micro_recall': tp/max(1, tp+fn),
            'micro_f1': 2*tp/max(1, 2*tp+fp+fn),
            'macro_precision': float(np.mean(per, axis=0)[0]),
            'macro_recall': float(np.mean(per, axis=0)[1]), 'macro_f1': float(np.mean(per, axis=0)[2]),
            'both_empty_count': empty, 'tp_fp_fn': counts.astype(int).tolist(),
            'empty_policy': 'Both empty: per-record P/R/F1=1; one empty: P/R/F1=0. Empty pairs add no micro counts.'}

def compute(rows):
    result = {}
    for group in ('overall', 'inclusion', 'exclusion'):
        part = rows if group == 'overall' else [r for r in rows if r['criterion_type'] == group]
        labels = list(LABELS.values()) if group == 'overall' else [LABELS[c] for c in ALLOWED[group]]
        gold = [r['gold_label'] for r in part]
        pred = [r['prediction']['label'] for r in part]
        result[group] = {'count': len(part), 'accuracy': accuracy_score(gold, pred),
                         'macro_f1': f1_score(gold, pred, labels=labels, average='macro', zero_division=0),
                         'per_class': classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0),
                         'confusion_labels': labels, 'confusion_matrix': confusion_matrix(gold, pred, labels=labels).tolist(),
                         'evidence': evidence_metrics(part)}
    return result
