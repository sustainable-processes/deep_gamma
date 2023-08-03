from dagster import op, graph, GraphOut, Out
from deep_gamma.ops import (
    lookup_smiles,
    resolve_smiles,
    find_clusters,
    cluster_split,  
    limit_outputs,
    plot_cluster_counts,
)
from deep_gamma.ops.split import merge_cluster_split
from deep_gamma.resources import (
    parquet_io_manager,
    csv_io_manager,
    pil_io_manager,
    mpl_io_manager,
    np_io_manager,
)
from deep_gamma import DATA_PATH
import pandas as pd




@graph
def cluster_split_data(molecule_list_df: pd.DataFrame, data: pd.DataFrame):
    """Split data into clusters and save split indices"""
    # Resolve data
    updated_molecule_list_df = lookup_smiles(molecule_list_df)
    new_data = resolve_smiles(data, updated_molecule_list_df)

    # Cluster data using molecule list
    clusters, clusters_df = find_clusters(updated_molecule_list_df)

    # Split data by clusters
    train_inds, valid_inds, test_inds = cluster_split(clusters, clusters_df)
    merge_cluster_split(train_inds, valid_inds, test_inds, clusters_df, new_data)

    # Visualize clusters
    plot_cluster_counts(clusters_df)


@op(out={"molecule_list_with_smiles": Out(), "data": Out()})
def dev_read_data_cosmo(context):
    """Dev mode read data in. Problably should change this to use Dagster modes"""
    context.log.info("Reading in data")
    data = pd.read_parquet(DATA_PATH / "02_intermediate" / "cosmo_data.pq")
    molecule_list_df = pd.read_csv(
        DATA_PATH / "01_raw" / "molecule_list.csv"
    )
    return molecule_list_df, data


@op(out={"molecule_list_with_smiles": Out(), "data": Out()})
def dev_read_data_aspen():
    """Dev mode read data in. Problably should change this to use Dagster modes"""
    data = pd.read_csv(DATA_PATH / "01_raw" / "aspen_data.csv")
    molecule_list_df = pd.Series(
        pd.concat(
            [data["smiles_1"], data["smiles_2"]]
        ).unique()
    ).to_frame().rename(columns={0: "smiles"})
    return molecule_list_df, data


@graph
def cluster_split_data_dev():
    """Just the cluster split"""
    molecule_list_df, data = dev_read_data_cosmo()
    new_data = limit_outputs(data)
    cluster_split_data(molecule_list_df,new_data)
    


### Graph for COSMO-RS data
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
        "model_input_np_io_manager": np_io_manager.configured(
            {
                "base_path": str(DATA_PATH / "05_model_input" / "cosmo"),
                "save_txt": True,
                "compress": False,
            }
        ),
        "model_input_csv_io_manager": csv_io_manager.configured(
            {"base_path": str(DATA_PATH / "05_model_input" / "cosmo")}
        ),
    },
    config={
        "ops": {
            "cluster_split_data": {
                "ops": {
                    "lookup_smiles": {
                        "config": dict(
                            input_column="solvent_name_1", smiles_column="smiles"
                        )
                    },
                    "resolve_smiles": {
                        "config": dict(
                            input_column_prefix="cas_number",
                            smiles_column_prefix="smiles",
                            molecule_df_input_column="cas_number",
                            molecule_df_smiles_column="smiles",
                            lookup_failed_smiles_as_name=False,
                            molecule_list_df_name_column="solvent_name_1",
                        ),
                    },

                    "find_clusters": {
                        "config": dict(
                            smiles_column="smiles",
                            cluster_column="cluster",
     
                        )
                    },
                    "plot_cluster_counts": {
                        "config": dict(cluster_column="cluster"),
                        "outputs": {
                            "result": {"filename": "cluster_counts.png", "dpi": 300}
                        },
                    },
                    "cluster_split": {"config": dict(valid_size=0.05, test_size=0.15)},
                    "merge_cluster_split": {"config": dict(subsample_valid_cont=0.01)},
                }
            },
            "limit_outputs": {
                "config": dict(min_value=-30.0, max_value=14.0)
            }
        },
        "loggers": {"console": {"config": {"log_level": "INFO"}}},
    },
)


# Graph for Aspen data
# csplit_job = cluster_split_data_dev.to_job(
#     resource_defs={
#         "intermediate_parquet_io_manager": parquet_io_manager.configured(
#             {"base_path": str(DATA_PATH / "02_intermediate")}
#         ),
#         "reporting_pil_io_manager": pil_io_manager.configured(
#             {"base_path": str(DATA_PATH / "08_reporting/aspen")}
#         ),
#         "reporting_mpl_io_manager": mpl_io_manager.configured(
#             {"base_path": str(DATA_PATH / "08_reporting/aspen")}
#         ),
#         "model_input_np_io_manager": np_io_manager.configured(
#             {
#                 "base_path": str(DATA_PATH / "05_model_input/aspen"),
#                 "save_txt": True,
#                 "compress": False,
#             }
#         ),
#         "model_input_csv_io_manager": csv_io_manager.configured(
#             {"base_path": str(DATA_PATH / "05_model_input/aspen")}
#         ),
#     },
#     config={
#         "ops": {
#             "cluster_split_data": {
#                 "ops": {
#                     "find_clusters": {
#                         # Tuned cutoff and min_cluster_size by hand to get
#                         # a good balance between train, validation and test.
#                         "config": dict(
#                             smiles_column="smiles",
#                             cutoff=0.6,
#                             cluster_column="cluster",
#                             min_cluster_size=2,
#                         )
#                     },
#                     "plot_cluster_counts": {
#                         "config": dict(cluster_column="cluster"),
#                         "outputs": {
#                             "result": {"filename": "cluster_counts.png", "dpi": 300}
#                         },
#                     },
#                     "cluster_split": {"config": dict(valid_size=0.05, test_size=0.05)},
#                     "merge_cluster_split": {"config": dict(
#                         subsample_valid_cont=0.01, features_columns=["x1", "TRange"]
#                     )}   
#                 }
#             },
#             "limit_outputs": {
#                 "config": dict(min_value=0.0, max_value=14.0)
#             }
#         },
#         "loggers": {"console": {"config": {"log_level": "INFO"}}},
#     },
# )

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
