from dagster import op, graph, GraphOut, InputDefinition, Out, ExperimentalWarning
from deep_gamma import USE_MODIN
from deep_gamma.ops import (
    lookup_smiles,
    resolve_smiles,
    get_scaffolds,
    scaffold_split,
    visualize_scaffolds,
    plot_scaffold_counts,
    find_clusters,
    plot_cluster_counts,
    cluster_split,
)
from deep_gamma.ops.split import merge_cluster_split
from deep_gamma.resources import (
    parquet_io_manager,
    parquet_loader,
    csv_loader,
    pil_io_manager,
    mpl_io_manager,
    np_io_manager,
)
from deep_gamma import DATA_PATH

if USE_MODIN:
    import modin.pandas as pd
else:
    import pandas as pd
from typing import Union
from pathlib import Path


@graph(
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
def cluster_split_data(molecule_list_df: pd.DataFrame, data: pd.DataFrame):
    # Cluster data using molecule list
    clusters, clusters_df = find_clusters(molecule_list_df)

    # Split data by clusters
    train_inds, valid_inds, test_inds = cluster_split(clusters, clusters_df)
    splits = merge_cluster_split(train_inds, valid_inds, test_inds, clusters_df, data)

    # Visualize clusters
    plot_cluster_counts(clusters_df)


@op(out={"molecule_list_with_smiles": Out(), "data": Out()})
def dev_read_data():
    data = pd.read_parquet(DATA_PATH / "02_intermediate" / "data_with_smiles.pq")
    molecule_list_df = pd.read_parquet(
        DATA_PATH / "02_intermediate" / "molecule_list_with_smiles.pq"
    )
    return molecule_list_df, data


@graph
def cluster_split_data_dev():
    molecule_list_df, data = dev_read_data()
    cluster_split_data(molecule_list_df, data)


@graph
def data_preprocessing():
    molecule_list_df, df = resolve_data()
    cluster_split_data(molecule_list_df)


# dp_job = data_preprocessing.to_job(
#     resource_defs={
#         "molecule_list_loader": csv_loader.configured(
#             {"path": str(DATA_PATH / "01_raw" / "molecule_list.csv")}
#         ),
#         "gamma_data_loader": parquet_loader.configured(
#             {
#                 "path": str(
#                     DATA_PATH / "02_intermediate" / "concatenated_cosmo_gammas.pq"
#                 )
#             }
#         ),
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
#     config={
#         "ops": {
#             "resolve_data": {
#                 "ops": {
#                     "resolve_smiles": {
#                         "config": dict(
#                             input_column_prefix="cas_number",
#                             smiles_column_prefix="smiles",
#                             molecule_df_input_column="cas_number",
#                             molecule_df_smiles_column="smiles",
#                             lookup_failed_smiles_as_name=True,
#                             molecule_list_df_name_column="cosmo_name",
#                         ),
#                     },
#                     "lookup_smiles": {
#                         "config": dict(
#                             input_column="cas_number", smiles_column="smiles"
#                         )
#                     },
#                 },
#             },
#             "cluster_split_data": {
#                 "ops": {
#                     "find_clusters": {
#                         "config": dict(
#                             smiles_column="smiles",
#                             cutoff=0.8,
#                             cluster_column="cluster",
#                             min_cluster_size=10,
#                         )
#                     },
#                     "plot_cluster_counts": {
#                         "config": dict(cluster_column="cluster"),
#                         "outputs": {
#                             "result": {"filename": "cluster_counts.png", "dpi": 300}
#                         },
#                     },
#                 }
#             },
#         }
#     },
# )


csplit_job = cluster_split_data_dev.to_job(
    resource_defs={
        "intermediate_parquet_io_manager": parquet_io_manager.configured(
            {"base_path": str(DATA_PATH / "02_intermediate")}
        ),
        "reporting_pil_io_manager": pil_io_manager.configured(
            {"base_path": str(DATA_PATH / "08_reporting")}
        ),
        "reporting_mpl_io_manager": mpl_io_manager.configured(
            {"base_path": str(DATA_PATH / "08_reporting")}
        ),
        "primary_np_io_manager": np_io_manager.configured(
            {
                "base_path": str(DATA_PATH / "03_primary"),
                "save_txt": True,
                "compress": True,
            }
        ),
    },
    config={
        "ops": {
            "cluster_split_data": {
                "ops": {
                    "find_clusters": {
                        "config": dict(
                            smiles_column="smiles",
                            cutoff=0.8,
                            cluster_column="cluster",
                            min_cluster_size=10,
                        )
                    },
                    "plot_cluster_counts": {
                        "config": dict(cluster_column="cluster"),
                        "outputs": {
                            "result": {"filename": "cluster_counts.png", "dpi": 300}
                        },
                    },
                    "cluster_split": {"config": dict(valid_size=0.05, test_size=0.05)},
                }
            },
        }
    },
)

if __name__ == "__main__":
    csplit_job.execute_in_process()
# def count_cluster(cutoff):
#     dists = []
#     nfps = len(fps)
#     for i in range(1, nfps):
#         sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
#         dists.extend([1 - x for x in sims])
#     scaffold_sets = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
#     scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))
#     scaffold_counts = [len(scaffold_set) for scaffold_set in scaffold_sets]
#     scaffold_counts = np.array(scaffold_counts)
#     frac_single_clusters = len(scaffold_counts[scaffold_counts < 2]) / len(
#         scaffold_counts
#     )
#     avg_cluster_size = np.mean(scaffold_counts)
#     std_cluster_size = np.std(scaffold_counts)
#     return frac_single_clusters, avg_cluster_size, std_cluster_size


# res = [count_cluster(0.1 * cutoff) for cutoff in range(10)]
# frac_single_clusters = [r[0] for r in res]
# avg_cluster_sizes = [r[1] for r in res]
# std_cluster_sizes = [r[2] for r in res]
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# fig.subplots_adjust(wspace=0.5)
# axes[0].scatter(np.arange(10) * 0.1, np.array(frac_single_clusters) * 100)
# axes[0].set_xlabel("Cutoff")
# axes[0].set_ylabel("Percentage of clusters with 1 molecule (%)")
# axes[1].errorbar(
#     np.arange(10) * 0.1,
#     np.array(avg_cluster_sizes),
#     yerr=std_cluster_sizes,
#     fmt="^",
#     linewidth=0,
#     elinewidth=1.0,
# )
# axes[1].set_xlabel("Cutoff")
# axes[1].set_ylabel("Average cluster size")
# fig.savefig("data/08_reporting/clustering_analysis.png", dpi=300)
