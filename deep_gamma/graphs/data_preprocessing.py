from dagster import op, graph, ExperimentalWarning
from deep_gamma.ops import (
    resolve_smiles,
    get_scaffolds,
    scaffold_split,
    visualize_scaffolds,
)
from deep_gamma import DATA_PATH
import pandas as pd
from typing import Union
from pathlib import Path


@op
def read_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH / "03_primary" / "cosmo_gammas_resolved.parquet")


@op
def read_molecule_list_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "03_primary" / "resolved_molecule_list.csv")


@op
def write_to_parquet(context, df: pd.DataFrame):
    df.to_parquet(DATA_PATH / "03_primary" / "cosmo_gammas_resolved.parquet")


@graph
def resolve_data():
    df = read_data()
    molecule_list_df = read_molecule_list_df()
    resolved_df = resolve_smiles(
        df,
        molecule_list_df,
    )
    write_to_parquet(resolved_df)


@graph
def scaffold_split_data():
    # Get data
    df = read_data()

    # Get scaffolds
    df, scaffolds = get_scaffolds(df)

    # Split based on scaffolds
    train_indices, valid_indices, test_indices = scaffold_split(df, scaffolds)

    # Visualize scaffolds
    train_img = visualize_scaffolds(df, scaffolds, train_indices)
    valid_img = visualize_scaffolds(df, scaffolds, valid_indices)
    test_img = visualize_scaffolds(df, scaffolds, test_indices)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    # resolve_data.execute_in_process(
    #     config={
    #         "resolve_smiles": {
    #             "config": dict(
    #                 input_column_prefix="cas_number",
    #                 smiles_column_prefix="smiles",
    #                 molecule_df_input_column="cas_number",
    #                 molecule_df_smiles_column="smiles",
    #                 lookup_failed_smiles_as_name=True,
    #                 molecule_list_df_name_column="cosmo_name",
    #             )
    #         }
    #     }
    # )
    # split_data.execute_in_process(
    #     config={"scaffold_split": {"config": dict(test_size=0.1)}}
    # )
    scaffold_split_data.execute_in_process(
        config={
            "get_scaffolds": {"config": dict(scaffold_columns=["smiles_1"])},
            "scaffold_split": {"config": dict(test_size=0.1, valid_size=0.05)},
        }
    )
