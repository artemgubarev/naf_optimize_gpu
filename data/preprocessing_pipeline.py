from datetime import datetime

import pytz
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.preprocessing import MinMaxScaler

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
from features.process_features import *

#from wlarf.features import *

import pandas as pd

df = pd.read_parquet('../../gsoc_incidents_raw3.parquet')

df['target'] = df['Вердикт'].apply(
    lambda x: True if x == 'False Positive' else (pd.NA if x == 'Не указан' else False)
)

dt = datetime(2024, 12, 1, tzinfo=pytz.UTC)  # datetime с UTC
df = df[df['Дата создания в источнике'] > dt]
df = df[df['target'].notnull()]
y = df['target'].astype(float).to_numpy()
used_columns = pd.read_csv('../src/features/used_columns.csv')
X = df[used_columns['column'].to_numpy()]
summary = pd.read_csv("../src/features/columns_summary.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
all_cols = set(X.columns)
inferred = summary.set_index('column')['inferred_type'].to_dict()

datetime_cols = [col for col in all_cols if inferred.get(col) == 'datetime']
time_cols = [col for col in all_cols if inferred.get(col) == 'time']
ip_cols = [col for col in all_cols if inferred.get(col) == 'ip/dns']
categorical_cols = [col for col in all_cols if inferred.get(col) == 'categorical']
object_cols = [col for col in all_cols if
               X_train[col].dtype == 'object' and col not in datetime_cols + time_cols + ip_cols + categorical_cols]

text_cols = [col for col in object_cols if col not in categorical_cols + ip_cols]
datetime_cols = list(set(datetime_cols + time_cols))
already_used = set(datetime_cols + ip_cols + categorical_cols + text_cols)
numeric_cols = [col for col in all_cols if col not in already_used and X_train[col].dtype.kind in 'iuf']

text_imputer = SimpleImputer(strategy='constant', fill_value="missing")
flatten_text = FunctionTransformer(flatten_text_func, validate=False)
to_str = FunctionTransformer(to_str_func)
ip_dns_transform = FunctionTransformer(transform_ip_dns, validate=False)
to_str_flatten = FunctionTransformer(to_str_flatten_func, validate=False)

datetime_pipeline = Pipeline([
    ('extract', FunctionTransformer(extract_datetime_features)),
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

# ip/dns
ip_transformers = [
    (f"ip_{col}", Pipeline([
        ("to_str", to_str),
        ('imputer', text_imputer),
        ('ip_dns_transform', ip_dns_transform),
        ('flatten', flatten_text),
        ('tfidf', TfidfVectorizer()),
        ('pca', PCA(n_components=100))
    ]), [col])
    for col in ip_cols
]

# Categorical
cat_transformers = [
    (f"сat_{col}", Pipeline([
        ("to_str", to_str),
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("rare_grouper", RareCategoryGrouper(threshold=0.01)),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), [col])
    for col in categorical_cols
]

# Numeric
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

# Text
text_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ('to_str_flatten', to_str_flatten),
    ("tfidf", TfidfVectorizer(max_features=50))
])

# Final ColumnTransformer
preprocessor = ColumnTransformer([
    ('datetime', datetime_pipeline, datetime_cols),
    ('numeric', num_pipeline, numeric_cols),
    *[(f"text_{col}", text_pipeline, [col]) for col in text_cols],
    *cat_transformers,
    *ip_transformers
])

# Final Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

pipeline.fit(X_train)

joblib.dump(pipeline, 'transform_data_pipeline.pkl')
