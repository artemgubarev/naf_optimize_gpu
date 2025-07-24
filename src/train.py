import sys
import os
import traceback
import psutil
import torch
import logging

logging.basicConfig(level=logging.DEBUG)

def print_mem(stage):
    print(f"\n--- {stage} ---")
    print(f"CPU: {psutil.virtual_memory().percent:.1f}% used")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src/features')))



import pandas as pd
import numpy as np
import warnings
import time

from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from process_features import preprocess_data
from forests import ForestKind, TaskType
from naf_model import NeuralAttentionForest, NAFParams

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

if __name__ == "__main__":
    try:
        warnings.filterwarnings("ignore")
        
        logfile = open('output.txt', 'w')
        sys.stdout = Tee(sys.stdout, logfile)


        # ----------- load data ---------------
        start_time = time.time()
        path = '../../gsoc_incidents_raw3.parquet'
        df = pd.read_parquet(path)

        df['target'] = df['Вердикт'].apply( lambda x: True if x == 'False Positive' else (pd.NA if x == 'Не указан' else False))
        df = df[df['target'].notnull()]
        df['target'] = df['target'].astype(float)
        df = df[::1]

        y = df['target'].astype(float).to_numpy()
        X = df.drop(columns=['target'])

        used_columns = pd.read_csv('../src/features/used_columns.csv')

        X = X[used_columns['column'].to_numpy()]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        X_train = preprocess_data(X_train, '../data/transform_data_pipeline.pkl')
        
        print(X.shape)


        # ----------- fit ---------------
        params = NAFParams(
            kind=ForestKind.RANDOM,
            task=TaskType.CLASSIFICATION,
            mode='end_to_end',
            loss='cross_entropy',
            weights_init_type='general_rule_normal',
            n_epochs=20,
            lr=0.001,
            lam=0.0,
            target_loss_weight=1.0,
            hidden_size=128,
            gpu=True,
            gpu_device = 3,
            n_layers=4,
            random_state=67890,
            forest=dict(
                n_estimators=200,
                min_samples_leaf=10,
                n_jobs=-1,
                max_depth=10,
                min_samples_split = 4,
            ))
        model = NeuralAttentionForest(params)

        start_time = time.time()
        print("⏳ start model fit...")
        model.fit(X_train, y_train)
        end_time = time.time()
        print("✅ model fit... %.2f sec" % (end_time - start_time))

        # ----------- optimize weights ---------------
        start_time = time.time()
        print("⏳ optimize weights...")
        model.optimize_weights(X_train, y_train, batch_size=2048, background_batch_size=1024)
        end_time = time.time()
        print("✅ optimize weights... %.2f sec" % (end_time - start_time))

        # ----------- predict ---------------
        start_time = time.time()
        print("⏳ predict...")
        X_test_proc = preprocess_data(X_test, '../data/transform_data_pipeline.pkl')
        y_proba = model.predict_batch(X_test_proc, batch_size=2048, background_batch_size=1024)[:, 1]

        thresholds = np.linspace(0, 1, 100)
        beta = 2
        f1_scores = []

        y_true = np.array([1 if label == 0 else 0 for label in y_test])
        for thr in thresholds:
            y_pred = (np.array([1 - score for score in y_proba]) >= 1 - thr).astype(int)
            f1 = fbeta_score(y_true, y_pred, beta=beta)
            f1_scores.append(f1)

        f1_scores = np.array(f1_scores)
        max_f1 = f1_scores.max()
        arg_f1 = f1_scores.argmax()

        end_time = time.time()
        print("✅ predict... %.2f sec" % (end_time - start_time))

        print(f'max F2 = {max_f1:.3f}, threshold = {arg_f1 / 100}')
        
        logfile.close()

    except Exception as e:
        print("\n🔥 EXCEPTION OCCURRED!")
        traceback.print_exc()