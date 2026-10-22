import numpy as np
import pandas as pd

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Pure Python/NumPy implementation matching scikit-learn's StandardScaler API.
    """
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        self.n_samples_seen_ = X_arr.shape[0]
        self.n_features_in_ = X_arr.shape[1]

        if self.with_mean:
            self.mean_ = np.mean(X_arr, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            self.var_ = np.var(X_arr, axis=0)
            scale = np.sqrt(self.var_)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.var_ = None
            self.scale_ = None

        return self

    def transform(self, X):
        if self.mean_ is None and self.scale_ is None:
            raise RuntimeError("This StandardScaler instance is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64, copy=self.copy)
        else:
            X_arr = np.array(X, dtype=np.float64, copy=self.copy)

        if self.with_mean:
            X_arr -= self.mean_
        if self.with_std:
            X_arr /= self.scale_
        return X_arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        X_arr = np.array(X, dtype=np.float64, copy=self.copy)
        if self.with_std:
            X_arr *= self.scale_
        if self.with_mean:
            X_arr += self.mean_
        return X_arr
