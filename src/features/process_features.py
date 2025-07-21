import os
import re

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.rare_categories_ = None

    def fit(self, X):
        counts = np.unique(X[:, 0], return_counts=True)
        unique = counts[0]
        freq = counts[1] / len(X)
        self.rare_categories_ = set(unique[freq < self.threshold])

        return self

    def transform(self, X):
        X_ = X.copy()
        X_[:, 0] = np.array(['rare' if x in self.rare_categories_ else x for x in X_[:, 0]])
        return X_


def extract_datetime_features(X):
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_datetime(X[col], errors='coerce')
        X[f"{col}_hour"] = X[col].dt.hour
        X[f"{col}_dayofweek"] = X[col].dt.dayofweek
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_weekend"] = X[col].dt.dayofweek.isin([5, 6]).astype(int)
        X = X.drop(columns=[col])

    return X


def split_ip_and_dns(s):
    parts = re.split(r'~~\|\|~~|[,\s]+', s)
    tokens = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.match(r'^\d+\.\d+\.\d+\.\d+$', p.split('|')[0]):
            ip = p.split('|')[0]
            tokens.extend([f"octet{i}_{part}" for i, part in enumerate(ip.split('.'))])
        else:
            if '(' in p:
                inner = re.findall(r'\(([^)]+)\)', p)
                for val in inner:
                    if re.match(r'^\d+\.\d+\.\d+\.\d+$', val):
                        tokens.extend([f"octet{i}_{part}" for i, part in enumerate(val.split('.'))])
            domain = p.split('|')[0].replace('\\', '').replace('(', '').replace(')', '')
            tokens.extend(domain.split('.'))
    return ' '.join(tokens)


def transform_ip_dns(X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.apply(split_ip_and_dns).values.reshape(-1, 1)
    elif isinstance(X, np.ndarray):
        return np.array([split_ip_and_dns(x[0] if x.ndim > 0 else x) for x in X]).reshape(-1, 1)
    else:
        raise TypeError(f"Unsupported input type: {type(X)}")


def flatten_text_func(x):
    return x.ravel()


def to_str_func(x):
    return x.astype(str)


def to_str_flatten_func(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.squeeze().astype(str).tolist()
    else:
        return [str(i) for i in x.ravel()]


def get_used_columns():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    used_columns_path = os.path.join(current_dir, 'used_columns.csv')

    used_columns = pd.read_csv(used_columns_path).squeeze().tolist()

    return used_columns


def preprocess_data(samples, pipeline: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    used_columns_path = os.path.join(current_dir, 'used_columns.csv')

    used_columns = pd.read_csv(used_columns_path)
    samples = samples[used_columns['column'].to_numpy()]

    transform_pipeline = joblib.load(pipeline)

    return transform_pipeline.transform(samples)
