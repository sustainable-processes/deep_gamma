"""This is a rough sketch of what I did to get the polynomial dataset v3"""
import pandas as pd
from pathlib import Path
import numpy as np


polynomial_df = pd.read_csv("data/03_primary/polynomial_good_fit.csv")
df = pd.read_csv("data_no_features.csv")
features = pd.read_csv("features.csv")
df = pd.concat([df, features], axis=1)

p = Path("data/05_model_input")
indices = {file.stem.rstrip("_indices"): np.loadtxt(file) for file in p.glob("*_indices.txt")}
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
for name, df_split in dfs_polynomials.items():
    df_split.to_csv(f"05_model_input/{name}_polynomial.csv", index=False)