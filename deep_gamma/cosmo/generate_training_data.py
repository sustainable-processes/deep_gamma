from sklearn.model_selection import train_test_split, GroupShuffleSplit
from chemprop.data import scaffold_to_smiles, generate_scaffold
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from pathlib import Path
import pkg_resources
from rdkit import Chem
from typing import List
from tqdm import trange


# Get parameters
step_params = {}
pipeline_params = {}
inputs = {}
molecule_df = pd.read_csv("data/01_raw/molecule_list.csv")

# Number of batches produced by ../data/cosmo/post_process_gammas.py
n_batches = step_params.get("n_batches", 191)

# Path to CSV files produced by ../data/cosmo/post_process_gammas.py
data_path  = pipeline_params.get("data_path", "data/01_raw/cosmo_batches/")
data_path = Path(data_path)

# Prefix of columns in CSV files that need to be resolved. Defaults to cas_number
input_column_prefix =  step_params.get("input_column_prefix","cas_number")

# Test size
test_size = step_params.get("test_size", 0.1)

# Get the name of the input column from the df outputed by the resolution step
molecule_df_input_column = inputs.get("molecule_df_input_column", "cas_number")



def main():
    dfs = [pd.read_csv(data_path / f"batch_{i}.csv") for i in trange(n_batches)]
    df = pd.concat(dfs)
    print(f"Total Rows: {df.shape[0]}")

    # Write out to file
    df.to_parquet("data/02_intermediate/cosmo_data.pq")

if __name__ == "__main__":
    main()
