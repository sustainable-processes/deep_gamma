from ast import For
from dagster import op, Out, Field, Output, Array
from itsdangerous import NoneAlgorithm
from deep_gamma import RecursiveNamespace

from chemprop.data import generate_scaffold
from rdkit import Chem


from sklearn.model_selection import GroupShuffleSplit, train_test_split
import pandas as pd
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
import logging
from typing import List, Tuple, Any, Dict
from tqdm.auto import tqdm


def scaffold_groups(mols: List[str]):
    """Find all the scaffolds and reference each one by index

    Parameters
    ---------
    mols: list of str
        The list of smiles strings

    Returns
    -------
    scaffolds, scaffold_indices
        scaffolds is a dictionary mapping scaffold to index.
        scaffold_indices gives the index of the scaffold for each molecule
    """
    scaffolds = dict()
    scaffold_num = 0
    scaffold_list = [""] * len(mols)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = scaffold_num
            scaffold_num += 1
        scaffold_list[i] = scaffold
    scaffold_indices = [scaffolds[s] for s in scaffold_list]
    return scaffolds, scaffold_indices


@op(
    config_schema=dict(
        scaffold_columns=Field(
            [str],
            description="List of columns that should be used for generating scaffolds",
        )
    ),
    out=dict(
        df_with_scaffolds=Out(
            pd.DataFrame, io_manager_key="intermediate_parquet_io_manager"
        ),
        scaffolds=Out(Dict[str, int]),
    ),
)
def get_scaffolds(context, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Get the scaffolds for all the unique molecules in in `scaffold_columns`"""
    scaffold_columns = context.solid_config["scaffold_columns"]

    # Get scaffolds
    unique_mols = np.concatenate([pd.unique(df[col]) for col in scaffold_columns])
    scaffolds, scaffold_indices = scaffold_groups(unique_mols)
    context.log.info(f"Number of scaffolds: {len(scaffolds)}")

    # Merge scaffolds
    scaffold_df = pd.DataFrame(
        {"scaffold_smiles": unique_mols, "scaffold_index": scaffold_indices}
    )
    for i, col in enumerate(scaffold_columns):
        df = df.merge(scaffold_df, left_on=col, right_on="scaffold_smiles", how="left")
        df = df.rename(columns={"scaffold_smiles": f"scaffold_smiles_{i}"})
    return df, scaffolds


@op(
    config_schema=dict(
        test_size=Field(float, description="Size of the test set"),
        valid_size=Field(float, description="Size of the validation set."),
    ),
    out=dict(
        train_indices=Out(np.ndarray),
        valid_indices=Out(np.ndarray),
        test_indices=Out(np.ndarray),
    ),
)
def scaffold_split(
    context, df: pd.DataFrame, scaffolds: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = RecursiveNamespace(**context.solid_config)

    # Train-test split
    splitter = GroupShuffleSplit(
        n_splits=2, test_size=config.test_size, random_state=1995
    )
    train_indices, test_indices = next(
        splitter.split(X=df, groups=df["scaffold_index"])
    )

    # Train-validation split
    valid_train_size = config.valid_size * len(df) / len(train_indices)
    train_df = df.iloc[train_indices]
    splitter = GroupShuffleSplit(
        n_splits=2, test_size=valid_train_size, random_state=1995
    )
    train_indices, valid_indices = next(
        splitter.split(X=train_df, groups=train_df["scaffold_index"])
    )

    # Logging
    context.log.info(f"Training set size: ~{len(train_indices)/1e6:.0f}M")
    context.log.info(f"Validation set size: ~{len(valid_indices)/1e3:.0f}k")
    context.log.info(f"Test set size: ~{len(test_indices)/1e3:.0f}k")
    null_scaffold = scaffolds[""]
    scaffold_counts = df["scaffold_index"].value_counts()
    context.log.info(
        f"Number of records that match the null scaffold: ~{scaffold_counts.loc[null_scaffold]/1e6:.0f}M"
    )
    return train_indices, valid_indices, test_indices


@op
def visualize_scaffolds(
    df: pd.DataFrame, scaffolds: dict, selection_indices: np.ndarray = None
):
    """Visualize scaffolds with counts of scaffolds below each molecule"""
    if selection_indices is not None:
        df = df.iloc[selection_indices]
    scaffold_counts = df["scaffold_index"].value_counts()
    null_scaffold = scaffolds[""]
    scaffold_idx_to_smiles = {idx: scaffold for scaffold, idx in scaffolds.items()}
    img = Chem.Draw.MolsToGridImage(
        [
            Chem.MolFromSmiles(scaffold_idx_to_smiles[idx])
            for idx in scaffold_counts.index
            if idx != null_scaffold
        ],
        molsPerRow=6,
        subImgSize=(200, 200),
        legends=[
            str(scaffold_counts.iloc[i])
            for i in range(len(scaffold_counts))
            if scaffold_counts.index[i] != null_scaffold
        ],
        returnPNG=False,
    )
    return img
