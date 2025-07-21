import sys
sys.path.append('NAF')
sys.path.append('LARF')
sys.path.append('LARF/wlarf/features')

import os
import re
import pytz
import joblib
import pandas as pd
import numpy as np
import wlarf
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from datetime import datetime

from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from IPython.core.debugger import set_trace

from process_features import preprocess_data
from naf.forests import ForestKind, TaskType
from naf.naf_model import NeuralAttentionForest, NAFParams
from naf import naf_original

import torch
from torch.profiler import profile, record_function, ProfilerActivity




# -------- data load ---------

print("⏳ data loading...")

path = 'LARF/gsoc_incidents_raw3.parquet'
df = pd.read_parquet(path)
df['target'] = df['Вердикт'].apply(
    lambda x: True if x == 'False Positive' else (pd.NA if x == 'Не указан' else False)
)
df = df[df['target'].notnull()]
df['target'] = df['target'].astype(float)
df = df[::2000]

y = df['target'].astype(float).to_numpy()
X = df.drop(columns=['target'])

used_columns = pd.read_csv('LARF/wlarf/features/used_columns.csv')

X = X[used_columns['column'].to_numpy()]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train = preprocess_data(X_train, 'LARF/transform_data_pipeline.pkl')

print("✅ data loaded successfully...")

params = NAFParams(
    kind=ForestKind.RANDOM,
    task=TaskType.CLASSIFICATION,
    mode='end_to_end',
    loss='cross_entropy',
    n_epochs=80,
    lr=0.01,
    lam=0.0,
    target_loss_weight=1.0,
    hidden_size=16,
    gpu=True,
    gpu_device = 3,
    n_layers=1,
    forest=dict(
        n_estimators=100,
        min_samples_leaf=1,
        n_jobs=-1
    ),
    random_state=67890
)

model = NeuralAttentionForest(params)
orig_model = naf_original.NeuralAttentionForest(params)

print("⏳ gpu model fit...")
model.fit(X_train, y_train)
print("⏳ gpu model optimize weights...")
model.optimize_weights(X_train, y_train)

print("⏳ original model fit...")
orig_model.fit(X_train, y_train)
print("⏳ original model optimize weights...")
orig_model.optimize_weights(X_train, y_train)

print("✅ all models fitted...")

X_test = preprocess_data(X_test, 'LARF/transform_data_pipeline.pkl')

def print_f2score(model, text):

    y_proba = model.predict(X_test)[:, 1]
    #y_proba = model.predict(X_test)
    thresholds = np.linspace(0, 1, 100)
    beta = 2

    max_f1_vals = []
    max_f1_args = []

    f1_scores = []

    y_true = np.array([1 if label == 0 else 0 for label in y_test])
    for thr in thresholds:
        y_pred = (np.array([1 - score for score in y_proba]) >= 1 - thr).astype(int)
        f1 = fbeta_score(y_true, y_pred, beta=beta)
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    max_f1 = f1_scores.max()
    arg_f1 = f1_scores.argmax()

    print(text)
    print(f'max F2 = {max_f1:.3f}, threshold = {arg_f1 / 100}')

print_f2score(model, "new model:\n")
print_f2score(orig_model, "original model:\n")