import numpy as np
import pandas as pd

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Pure python train_test_split implementation matching scikit-learn."""
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
        
    n_samples = len(arrays[0])
    for a in arrays:
        if len(a) != n_samples:
            raise ValueError("All input arrays must have the same length")

    if test_size is None and train_size is None:
        test_size = 0.25
    if test_size is not None and isinstance(test_size, float):
        n_test = int(np.round(n_samples * test_size))
    elif test_size is not None:
        n_test = int(test_size)
    else:
        n_test = n_samples - int(np.round(n_samples * train_size))

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)

    if shuffle:
        if stratify is not None:
            train_idx, test_idx = [], []
            classes, y_indices = np.unique(stratify, return_inverse=True)
            for cls_idx in range(len(classes)):
                cls_samples = indices[y_indices == cls_idx]
                rng.shuffle(cls_samples)
                cls_n_test = int(np.round(len(cls_samples) * (n_test / n_samples)))
                test_idx.extend(cls_samples[:cls_n_test])
                train_idx.extend(cls_samples[cls_n_test:])
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
        else:
            rng.shuffle(indices)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
    else:
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    result = []
    for a in arrays:
        if isinstance(a, (pd.DataFrame, pd.Series)):
            result.append(a.iloc[train_idx])
            result.append(a.iloc[test_idx])
        else:
            arr = np.asarray(a)
            result.append(arr[train_idx])
            result.append(arr[test_idx])
    return result
