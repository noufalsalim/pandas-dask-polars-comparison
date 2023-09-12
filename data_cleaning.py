# Code checked on python version 3.10.2

# Version used in code
# dask==2023.8.1
# pandas==2.0.3
# polars==0.18.15

import pandas as pd
import dask.dataframe as dd
import polars as pl

# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Pandas Data Cleaning and Transformation
def pandas_clean_transform():
    df_pandas = pd.read_csv(url)
    
    # Remove rows with missing values
    df_pandas.dropna(inplace=True)
    
    # Convert species names to uppercase
    df_pandas['species'] = df_pandas['species'].str.upper()
    
    print("Pandas Cleaned and Transformed DataFrame:")
    print(df_pandas.head())

# Dask Data Cleaning and Transformation
def dask_clean_transform():
    df_dask = dd.read_csv(url)
    
    # Remove rows with missing values
    df_dask = df_dask.dropna()
    
    # Convert species names to uppercase
    df_dask['species'] = df_dask['species'].str.upper()
    
    print("Dask Cleaned and Transformed DataFrame:")
    print(df_dask.head())

# Polars Data Cleaning and Transformation
def polars_clean_transform():
    df_polars = pl.read_csv(url)
    
    # Remove rows with missing values
    df_polars = df_polars.drop_nulls()
    
    # Convert species names to uppercase
    df_polars = df_polars.with_columns(pl.col("species").str.to_uppercase())
    
    print("Polars Cleaned and Transformed DataFrame:")
    print(df_polars.head())

# Perform data cleaning and transformation for each framework
pandas_clean_transform()
print("\n---------------------------------\n")
dask_clean_transform()
print("\n---------------------------------\n")
polars_clean_transform()