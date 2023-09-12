# Code checked on python version 3.10.2

# Version used in code
# dask==2023.8.1
# dask-ml==2023.3.24
# pandas==2.0.3
# polars==0.18.15
# scikit-learn==1.3.0

import pandas as pd
import dask.dataframe as dd
import polars as pl
from dask_ml.model_selection import train_test_split as tts
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Pandas Machine Learning Pipeline
def pandas_ml_pipeline():
    df_pandas = pd.read_csv(url)
    
    # Data Preprocessing
    X = df_pandas.drop('species', axis=1)
    y = df_pandas['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Pandas ML Pipeline Accuracy:", accuracy)

# Dask Machine Learning Pipeline
def dask_ml_pipeline():
    df_dask = dd.read_csv(url)
    
    # Data Preprocessing
    X = df_dask.drop('species', axis=1)
    y = df_dask['species']
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)
    
    # Model Training (Dask supports scikit-learn's API)
    model = RandomForestClassifier()
    model.fit(X_train.compute(), y_train.compute())
    
    # Model Evaluation
    y_pred = model.predict(X_test.compute())
    accuracy = accuracy_score(y_test.compute(), y_pred)
    
    print("Dask ML Pipeline Accuracy:", accuracy)

# Polars Machine Learning Pipeline
def polars_ml_pipeline():
    df_polars = pl.read_csv(url)
    
    # Data Preprocessing
    X = df_polars.drop("species")
    y = df_polars.select("species")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training (Polars supports scikit-learn's API)
    model = RandomForestClassifier()
    model.fit(X_train, y_train.with_columns(pl.col("species").flatten()))
    
    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Polars ML Pipeline Accuracy:", accuracy)

# Perform machine learning pipelines for each framework
pandas_ml_pipeline()
print("\n---------------------------------\n")
dask_ml_pipeline()
print("\n---------------------------------\n")
polars_ml_pipeline()