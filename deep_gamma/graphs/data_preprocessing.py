from dagster import op, graph, GraphOut, InputDefinition, ExperimentalWarning
from deep_gamma.ops import (
    lookup_smiles,
    resolve_smiles,
    get_scaffolds,
    scaffold_split,
    visualize_scaffolds,
    plot_scaffold_counts,
    find_clusters,
)
from deep_gamma.resources import (
    parquet_io_manager,
    parquet_loader,
    csv_loader,
    pil_io_manager,
    mpl_io_manager,
)
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


@graph(
    # input_defs=[
    #     InputDefinition("molecule_list_df", root_manager_key="molecule_list_loader"),
    #     InputDefinition("data", root_manager_key="gamma_data_loader"),
    # ],
    out=dict(molecule_list_df=GraphOut(), data=GraphOut()),
)
def resolve_data():
    """Resolve data to find all the SMILES strings"""
    # Lookup missing SMILES
    molecule_list_df = lookup_smiles()

    # Resolve SMILES
    df = resolve_smiles(molecule_list_df=molecule_list_df)

    return {"molecule_list_df": molecule_list_df, "data": df}


@graph
def scaffold_split_data(df: pd.DataFrame):
    """Split data based on scaffods"""
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
    plot_scaffold_counts(df)


@graph
def cluster_split_data(molecule_list_df: pd.DataFrame):
    # Cluster data using molecule list
    clusters = find_clusters(molecule_list_df)

    # Split data by clusters
    # split_data()

    # Visualize clusters
    # visualize_clusters(0)


@graph
def data_preprocessing():
    molecule_list_df, df = resolve_data()
    cluster_split_data(molecule_list_df)


dp_job = data_preprocessing.to_job(
    resource_defs={
        "molecule_list_loader": csv_loader.configured(
            {"path": str(DATA_PATH / "01_raw" / "molecule_list.csv")}
        ),
        "gamma_data_loader": parquet_loader.configured(
            {
                "path": str(
                    DATA_PATH / "02_intermediate" / "concatenated_cosmo_gammas.pq"
                )
            }
        ),
        "intermediate_parquet_io_manager": parquet_io_manager.configured(
            {"base_path": str(DATA_PATH / "02_intermediate")}
        ),
        "reporting_pil_io_manager": pil_io_manager.configured(
            {"base_path": str(DATA_PATH / "08_reporting")}
        ),
        "reporting_mpl_io_manager": mpl_io_manager.configured(
            {"base_path": str(DATA_PATH / "08_reporting")}
        ),
    },
    config={
        "ops": {
            "resolve_data": {
                "ops": {
                    "resolve_smiles": {
                        "config": dict(
                            input_column_prefix="cas_number",
                            smiles_column_prefix="smiles",
                            molecule_df_input_column="cas_number",
                            molecule_df_smiles_column="smiles",
                            lookup_failed_smiles_as_name=True,
                            molecule_list_df_name_column="cosmo_name",
                        ),
                    },
                    "lookup_smiles": {
                        "config": dict(
                            input_column="cas_number", smiles_column="smiles"
                        )
                    },
                },
            },
            "cluster_split_data": {
                "ops": {
                    "find_clusters": {
                        "config": dict(
                            smiles_column="smiles",
                            cutoff=0.8,
                            cluster_column="cluster",
                        )
                    }
                }
            },
        }
    },
)


# dp_job = data_preprocessing.to_job(
#     resource_defs={
#         "intermediate_parquet_io_manager": parquet_io_manager.configured(
#             {"base_path": str(DATA_PATH / "02_intermediate")}
#         ),
#         "reporting_pil_io_manager": pil_io_manager.configured(
#             {"base_path": str(DATA_PATH / "08_reporting")}
#         ),
#         "reporting_mpl_io_manager": mpl_io_manager.configured(
#             {"base_path": str(DATA_PATH / "08_reporting")}
#         ),
#     },
#     # config={
#     #     "ops": {
#     #         {
#     #             "resolve_smiles": {
#     #                 "config": dict(
#     #                     input_column_prefix="cas_number",
#     #                     smiles_column_prefix="smiles",
#     #                     molecule_df_input_column="cas_number",
#     #                     molecule_df_smiles_column="smiles",
#     #                     lookup_failed_smiles_as_name=True,
#     #                     molecule_list_df_name_column="cosmo_name",
#     #                 )
#     #             },
#     #             "find_clusters": {
#     #                 "config": dict(
#     #                     smiles_column="smiles", cutoff=0.8, cluster_column="cluster"
#     #                 )
#     #             },
#     #         }
#     #     }
#     # },
# )
