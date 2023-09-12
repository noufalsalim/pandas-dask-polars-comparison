# Code checked on python version 3.10.2

# Version used in code
# dask==2023.8.1
# pandas==2.0.3
# polars==0.18.15

import pandas as pd
import dask.dataframe as dd
import polars as pl

# Load the Magestic million dataset, this data is used since it has 1 million rows
# and much more suited for big data code
url = "./majestic_million.csv"

# Pandas Big Data Analytics (Optimized)
def pandas_big_data_analytics():
    chunk_size = 5000
    chunks = pd.read_csv(url, chunksize=chunk_size)
    
    # Initialize result placeholders
    avg_global_rank = pd.Series()
    filtered_data = pd.DataFrame()
    max_global_rank = pd.Series()
    
    for chunk in chunks:
        # Group by TLD and compute average Global Rank
        avg_global_rank = pd.concat([avg_global_rank, chunk.groupby('TLD')['GlobalRank'].mean()])
        
        # Filter rows with global rank equals to previous global rank
        filtered_data = pd.concat([filtered_data, chunk['GlobalRank'] == chunk['PrevGlobalRank']])
        
        # Calculate the maximum global rank for each TLD
        max_global_rank = pd.concat([max_global_rank, chunk.groupby('TLD')['GlobalRank'].max()])
    
    print("Pandas Big Data Analytics Results:")
    print("Average Global Rank by TLD:\n", avg_global_rank.groupby(level=0).mean())
    print("Filtered Data (global rank = previous global rank):\n", filtered_data)
    print("Maximum Global Rank by TLD:\n", max_global_rank.groupby(level=0).max())

# Dask Big Data Analytics
def dask_big_data_analytics():
    df_dask = dd.read_csv(url)
    
    # Group by TLD and compute average Global Rank
    avg_global_rank = df_dask.groupby('TLD')['GlobalRank'].mean().compute()
    
    # Filter rows with global rank equals to previous global rank
    filtered_data = df_dask[df_dask['GlobalRank'] == df_dask['PrevGlobalRank']].compute()
    
    # Calculate the maximum global rank for each TLD
    max_global_rank = df_dask.groupby('TLD')['GlobalRank'].max().compute()
    
    print("Dask Big Data Analytics Results:")
    print("Average Global Rank by TLD:\n", avg_global_rank)
    print("Filtered Data (global rank = previous global rank):\n", filtered_data)
    print("Maximum Global Rank by TLD:\n", max_global_rank)


# Polars Big Data Analytics (Optimized)
def polars_big_data_analytics():
    df_polars = pl.read_csv(url)
    
    # Group by TLD and compute average Global Rank
    avg_global_rank = df_polars.groupby('TLD').agg(pl.mean('GlobalRank'))
    
    # Filter rows with global rank equals to previous global rank
    filtered_data = df_polars.filter(df_polars['GlobalRank'] == df_polars['PrevGlobalRank'])
    
    # Calculate the maximum global rank for each TLD
    max_global_rank = df_polars.groupby('TLD').agg(pl.max('GlobalRank'))
    
    print("Polars Big Data Analytics Results:")
    print("Average Global Rank by TLD:\n", avg_global_rank)
    print("Filtered Data (global rank = previous global rank):\n", filtered_data)
    print("Maximum Global Rank by TLD:\n", max_global_rank)


# Perform big data analytics using Pandas (Optimized)
pandas_big_data_analytics()
print("\n---------------------------------\n")
# Perform big data analytics using Dask
dask_big_data_analytics()
print("\n---------------------------------\n")
# Perform big data analytics using Polars (Optimized)
polars_big_data_analytics()