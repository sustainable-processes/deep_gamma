"""This is a rough sketch of what I did to get the polynomial dataset v3"""
import pandas as pd
from pathlib import Path
import numpy as np


polynomial_df = pd.read_csv("data/03_primary/polynomial_good_fit.csv")
df = pd.read_csv("data/data_no_features.csv")
features = pd.read_csv("data/features.csv")
df = pd.concat([df, features], axis=1)

p = Path("data/05_model_input")
indices = {file.stem.rpartition("_indices")[0]: np.loadtxt(file) for file in p.glob("*_indices.txt")}
dfs = {
    name: df.iloc[inds].drop(
        ["ln_gamma_1", "ln_gamma_2", "x(1)"], axis=1
    ) 
    for name, inds in indices.items()
}
dfs_polynomials = {
    name: polynomial_df.merge(
        df_split, 
        on=["smiles_1", "smiles_2", "temperature (K)"],
        how="inner"
    ).drop_duplicates()
    for name, df_split in dfs.items()
}

main_columns = [
    "smiles_1","smiles_2",
    "c0_0","c1_0","c2_0","c3_0","c4_0",
    "c0_1","c1_1","c2_1","c3_1","c4_1"
]
for name, df_split in dfs_polynomials.items():
    df_split[main_columns].to_csv(f"data/05_model_input/{name}_polynomial.csv", index=False)
    df_split["temperature (K)"].to_csv(f"data/05_model_input/{name}_polynomial_temperature.csv", index=False)