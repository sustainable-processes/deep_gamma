from dagster import op, graph, ExperimentalWarning
from deep_gamma.ops import (
    resolve_smiles,
    get_scaffolds,
    scaffold_split,
    visualize_scaffolds,
)
from deep_gamma.resources import parquet_io_manager, pil_io_manager
from deep_gamma import DATA_PATH
import pandas as pd
from typing import Union
from pathlib import Path


@op
def read_data() -> pd.DataFrame:
    return pd.read_parquet(
        DATA_PATH / "02_intermediate" / "cosmo_gammas_resolved.parquet"
    )


@op
def read_molecule_list_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "03_primary" / "resolved_molecule_list.csv")


@op
def write_to_parquet(context, df: pd.DataFrame):
    df.to_parquet(DATA_PATH / "03_primary" / "cosmo_gammas_resolved.parquet")


@graph
def resolve_data():
    """Resolve data to find all the SMILES strings"""
    # Read in data
    df = read_data()
    molecule_list_df = read_molecule_list_df()

    # Resolve SMILES
    resolved_df = resolve_smiles(
        df,
        molecule_list_df,
    )

    # Resolve SMILES
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
    vs_train = visualize_scaffolds.alias("visualize_scaffolds_train")
    vs_valid = visualize_scaffolds.alias("visualize_scaffolds_valid")
    vs_test = visualize_scaffolds.alias("visualize_scaffolds_test")
    vs_train(df, scaffolds, train_indices)
    vs_valid(df, scaffolds, valid_indices)
    vs_test(df, scaffolds, test_indices)


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
    job = scaffold_split_data.to_job(
        resource_defs={
            "intermediate_parquet_io_manager": parquet_io_manager.configured(
                {"base_path": str(DATA_PATH / "02_intermediate")}
            ),
            "reporting_pil_io_manager": pil_io_manager.configured(
                {"base_path": str(DATA_PATH / "08_reporting")}
            ),
        },
        config={
            "ops": {
                "get_scaffolds": {"config": dict(scaffold_columns=["smiles_1"])},
                "scaffold_split": {"config": dict(test_size=0.1, valid_size=0.05)},
                "visualize_scaffolds_train": {
                    "outputs": {"result": {"filename": "train_scaffolds.png"}}
                },
                "visualize_scaffolds_valid": {
                    "outputs": {"result": {"filename": "valid_scaffolds.png"}}
                },
                "visualize_scaffolds_test": {
                    "outputs": {"result": {"filename": "test_scaffolds.png"}}
                },
            }
        },
    )
    job.execute_in_process()
