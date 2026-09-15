"""Tie-aware arrival ranking; thresholds are development diagnostics only."""
import numpy as np


def precision_recall(scores, truth):
    scores, truth = np.asarray(scores), np.asarray(truth, dtype=bool)
    order = np.argsort(-scores, kind='stable')
    scores, truth = scores[order], truth[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    tp = np.cumsum(truth)[ends]
    precision = tp / (ends+1)
    recall = tp / max(int(truth.sum()), 1)
    ap = float(np.sum(np.diff(np.r_[0., recall])*precision)) if truth.any() else None
    return dict(average_precision=ap, prevalence=float(truth.mean()),
                thresholds=scores[ends].tolist(), precision=precision.tolist(), recall=recall.tolist())


def ranking_report(probabilities, labels):
    p, y = np.asarray(probabilities), np.asarray(labels)
    return {str(m): precision_recall(p[:, :m//5].sum(1), y < m//5) for m in (15,30,60)}


def plot_ranking(report):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for ax, (h, r) in zip(axes, report.items()):
        ax.step(r['recall'], r['precision'], where='pre', label=f"AP={r['average_precision']}")
        ax.axhline(r['prevalence'], color='gray', linestyle='--', label='Constant score')
        ax.set(xlabel='Recall', ylabel='Precision', title=f'Arrival within {h} minutes', xlim=(0,1), ylim=(0,1))
        ax.legend()
    fig.tight_layout(); plt.show()
