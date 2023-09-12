# Code checked on python version 3.10.2

# Version used in code
# dask==2023.8.1
# pandas==2.0.3
# polars==0.18.15

# Import necessary libraries
import pandas as pd
import dask.dataframe as dd
import polars as pl

# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Pandas EDA
def pandas_eda():
    df_pandas = pd.read_csv(url)
    
    print("Pandas Basic EDA:")
    print("Number of rows:", len(df_pandas))
    print("Columns:", df_pandas.columns)
    print("Summary Statistics:\n", df_pandas.describe())
    print("Unique Species:", df_pandas['species'].unique())

# Dask EDA
def dask_eda():
    df_dask = dd.read_csv(url)
    
    print("Dask Basic EDA:")
    print("Number of rows:", len(df_dask))
    print("Columns:", df_dask.columns)
    print("Summary Statistics:\n", df_dask.describe().compute())
    print("Unique Species:", df_dask['species'].unique().compute())

# Polars EDA
def polars_eda():
    df_polars = pl.read_csv(url)
    
    print("Polars Basic EDA:")
    print("Number of rows:", len(df_polars))
    print("Columns:", df_polars.columns)
    print("Summary Statistics:\n", df_polars.describe())
    print("Unique Species:", df_polars.select("species").unique())

# Perform EDA for each framework
pandas_eda()
print("\n---------------------------------\n")
dask_eda()
print("\n---------------------------------\n")
polars_eda()