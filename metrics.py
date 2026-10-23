"""Jet-tagging metrics, same definitions as the ParT paper (arXiv:2202.03772 Sec. 2)."""
import numpy as np

# signal classes vs. QCD background, and the signal efficiency (TPR) used
# for the background-rejection metric RejX% = 1 / FPR at TPR = X%
REJ_TARGETS = [("Hbb", 0.5), ("Hcc", 0.5), ("Hgg", 0.5), ("H4q", 0.5), ("Hqql", 0.99),
               ("Tbqq", 0.5), ("Tbl", 0.995), ("Wqq", 0.5), ("Zqq", 0.5)]


def compute_jet_tagging_metrics(scores, labels, class_names):
    """scores: (n, C) softmax outputs; labels: (n,) ints."""
    from sklearn.metrics import roc_auc_score, roc_curve

    out = {}
    preds = scores.argmax(1)
    out["accuracy"] = float((preds == labels).mean())
    try:
        out["auc_macro_ovo"] = float(roc_auc_score(labels, scores, average="macro",
                                                   multi_class="ovo"))
    except ValueError:
        out["auc_macro_ovo"] = float("nan")

    if "QCD" in class_names:
        b = class_names.index("QCD")
        rej = {}
        for name, tpr_target in REJ_TARGETS:
            if name not in class_names:
                continue
            s_idx = class_names.index(name)
            sel = (labels == s_idx) | (labels == b)
            y = (labels[sel] == s_idx).astype(int)
            if y.sum() == 0 or (1 - y).sum() == 0:
                continue
            sc = scores[sel, s_idx] / (scores[sel, s_idx] + scores[sel, b] + 1e-12)
            fpr, tpr, _ = roc_curve(y, sc)
            fpr_at = float(np.interp(tpr_target, tpr, fpr))
            rej[f"{name}_Rej{tpr_target * 100:g}"] = (1.0 / fpr_at) if fpr_at > 0 else float("inf")
        out["rejection"] = rej
    return out
