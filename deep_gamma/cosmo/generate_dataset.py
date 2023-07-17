import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
from pathlib import Path
from rdkit import Chem
from typing import List


# Get parameters
step_params = {}
pipeline_params = {}
inputs = {}
molecule_df = pd.read_csv("../data/cosmo/solvent_descriptors.csv")

# Number of batches produced by ../data/cosmo/post_process_gammas.py
n_batches = step_params.get("n_batches", 191)

# Path to CSV files produced by ../data/cosmo/post_process_gammas.py
data_path  = pipeline_params.get("data_path", "../data/cosmo/cosmo_batches/")
data_path = Path(data_path)

# Prefix of columns in CSV files that need to be resolved. Defaults to cas_number
input_column_prefix =  step_params.get("input_column_prefix","cas_number")

# Test size
test_size = step_params.get("test_size", 0.1)

# Get the name of the input column from the df outputed by the resolution step
molecule_df_input_column = inputs.get("molecule_df_input_column", "cas_number")

# Resolve to SMILES by merging
def resolve_smiles(df, molecule_df):
    df = df.drop("Unnamed: 0", axis=1)
    new_df = df.copy()
    drop_columns = molecule_df.columns.tolist()
    drop_columns.remove("smiles")
    for i in [1,2]:
        new_df = pd.merge(
            new_df,
            molecule_df,
            left_on=f"{input_column_prefix}_{i}",
            right_on=molecule_df_input_column, 
            how="left"
        )
        new_df = new_df.rename(columns={"smiles": f"smiles_{i}"}).drop(drop_columns, axis=1)

        # Deal with the molecule_df that don't have a CAS number
        # The cosmo name was used instead
        names = (df
                 .mask(df[f"cas_number_{i}"]
                 .str.contains(r"\b[1-9]{1}[0-9]{1,5}-\d{2}-\d\b", regex=True))
                 .reset_index()
                 .dropna()
                 .drop("index", axis=1)
                 .reset_index()
        )
        smiles_names = pd.merge(names, molecule_df, left_on=f"cas_number_{i}", right_on="cosmo_name").set_index("index")
        smiles_names = smiles_names[["smiles"]].rename(columns={"smiles": f"smiles_{i}"})
        new_df.update(smiles_names)
        new_df =  new_df.dropna()
    return new_df

dfs = []
for i in range(n_batches):
    print(get_mem())
    print(f"Reading in COMSO Data: batch {i+1} of {n_batches}")
    df = pd.read_csv(data_path / f"batch_{i}.csv")
    print(f"Resolving batch {i+1} of {n_batches}")
    new_df = resolve_smiles(df, molecule_df)
    dfs.append(new_df)
    clear_output(wait=True)
df = pd.concat(dfs)
print(f"Total Rows: {df.shape[0]}")

print(f"Total Rows: ~{df.shape[0]/1e6:.0f}M")

# Save intermediate data in case a failure occurs
df.to_csv("../data/cosmo/cosmo_data.csv")