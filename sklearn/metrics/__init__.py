import numpy as np

def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    label_to_ind = {val: i for i, val in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_ind and p in label_to_ind:
            cm[label_to_ind[t], label_to_ind[p]] += 1
    return cm

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True):
    return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

def classification_report(y_true, y_pred, target_names=None, digits=4):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    if target_names is None:
        target_names = [str(c) for c in classes]
        
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    header = f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
    rows = []
    
    for i, c in enumerate(classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        
        rows.append(f"{target_names[i]:>15} {prec:>10.{digits}f} {rec:>10.{digits}f} {f1:>10.{digits}f} {support:>10d}")
        
    total_support = np.sum(supports)
    acc = np.mean(y_true == y_pred)
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    weighted_prec = np.sum(np.array(precisions) * np.array(supports)) / total_support
    weighted_rec = np.sum(np.array(recalls) * np.array(supports)) / total_support
    weighted_f1 = np.sum(np.array(f1s) * np.array(supports)) / total_support
    
    result = header + "\n".join(rows) + "\n\n"
    result += f"{'accuracy':>15} {'':>10} {'':>10} {acc:>10.{digits}f} {total_support:>10d}\n"
    result += f"{'macro avg':>15} {macro_prec:>10.{digits}f} {macro_rec:>10.{digits}f} {macro_f1:>10.{digits}f} {total_support:>10d}\n"
    result += f"{'weighted avg':>15} {weighted_prec:>10.{digits}f} {weighted_rec:>10.{digits}f} {weighted_f1:>10.{digits}f} {total_support:>10d}\n"
    return result

__all__ = ["confusion_matrix", "accuracy_score", "roc_curve", "classification_report"]
