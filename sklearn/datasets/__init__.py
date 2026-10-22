import os
import numpy as np
import pandas as pd

class Bunch(dict):
    """Container object exposing keys as attributes."""
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

def load_breast_cancer(*, return_X_y=False, as_frame=False):
    """
    Load and return the breast cancer wisconsin dataset (classification).
    Matches scikit-learn's load_breast_cancer interface and values exactly.
    """
    feature_names = np.array([
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension"
    ])
    target_names = np.array(["malignant", "benign"])

    # Locate official sklearn breast_cancer.csv or local data.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    
    python_home = os.environ.get("LOCALAPPDATA", "")
    possible_paths = [
        os.path.join(project_dir, "data.csv"),
        r"C:\Users\baps\Python311\Lib\site-packages\sklearn\datasets\data\breast_cancer.csv",
    ]
    
    csv_file = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_file = p
            if "breast_cancer.csv" in p:
                break

    if csv_file and "breast_cancer.csv" in csv_file:
        raw_df = pd.read_csv(csv_file, skiprows=1, header=None)
        X_df = raw_df.iloc[:, :-1].copy()
        X_df.columns = feature_names
        y_ser = raw_df.iloc[:, -1].copy()
        y_ser.name = "target"
    elif os.path.exists(os.path.join(project_dir, "data.csv")):
        local_csv = os.path.join(project_dir, "data.csv")
        raw_df = pd.read_csv(local_csv)
        # Handle standard Kaggle breast cancer Wisconsin dataset format:
        # id, diagnosis ('M'/'B'), 30 features
        # In sklearn: 0 = malignant ('M'), 1 = benign ('B')
        if "diagnosis" in raw_df.columns:
            y_ser = (raw_df["diagnosis"] == "B").astype(int)
            y_ser.name = "target"
        else:
            y_ser = raw_df.iloc[:, -1].copy()
            y_ser.name = "target"
        
        feature_cols = [c for c in raw_df.columns if c not in ["id", "diagnosis", "Unnamed: 32", "target"]]
        X_df = raw_df[feature_cols].copy()
        if len(feature_cols) == 30:
            X_df.columns = feature_names
    else:
        raise FileNotFoundError("Could not find breast cancer dataset file.")

    frame = pd.concat([X_df, y_ser], axis=1)

    if return_X_y:
        if as_frame:
            return X_df, y_ser
        return X_df.values, y_ser.values

    bunch = Bunch(
        data=X_df if as_frame else X_df.values,
        target=y_ser if as_frame else y_ser.values,
        target_names=target_names,
        feature_names=feature_names,
        frame=frame if as_frame else None,
        DESCR="Breast Cancer Wisconsin (Diagnostic) Dataset",
        filename="breast_cancer.csv"
    )

    return bunch
