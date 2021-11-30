"""
3 prediction cases
Both molecules are trained on but now at a different temperature  / composition (easiest)
Only trained on one of the molecules (medium)
Not trained on either molecule (hardest)

Three validation / test sets
Leave out some compositions and temperatures during training
Only one molecule in training set
Both molecule not in training set

Approach
1. Randomly split molecule list into train and valid
2. Valid CONT: Just training molecules at temperatures and compositions not test
3. Valid MIX: All combinations of training molecules and validation molecules
4. Valid INDP: Just combinations of molecules in the validation set
5. Do 2-4 for test set as well

"""

from deep_gamma import RecursiveNamespace, USE_MODIN
from chemprop.data import generate_scaffold
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dagster import DynamicOut, op, Out, Field, Output, Array
from PIL.Image import Image
from tqdm.auto import tqdm
from typing import List, Tuple, Dict
from tqdm.auto import tqdm
import os


@op(
    config_schema=dict(
        smiles_column=Field(
            str,
            description="Column containing SMILES strings to use for forming clusters",
            default_value="smiles",
        ),
        cutoff=Field(
            float,
            description="The cutoff value for tanimoto similarity.  Molecules that are more similar than this will tend to be put in the same dataset.",
        ),
        min_cluster_size=Field(
            int,
            description="Minimimum size of clusters. Any clusters smaller than this size are combined with the smallest cluster above the minimum cluster size.",
        ),
        cluster_column=Field(
            str,
            description="Name of the column to create with the clusters",
            default_value="cluster",
        ),
    ),
    out=dict(
        clusters=Out(),
        molecule_list_with_clusters=Out(
            io_manager_key="intermediate_parquet_io_manager"
        ),
    ),
)
def find_clusters(context, df: pd.DataFrame) -> Tuple[List, pd.DataFrame]:
    """Find clusters using the Butina algorithm"""
    config = RecursiveNamespace(**context.solid_config)
    # Create fingerprints
    mols = []
    for smiles in df[config.smiles_column]:
        try:
            mols.append(Chem.MolFromSmiles(smiles))
        except TypeError:
            context.log.error(f"Could not convert {smiles} to RDKit molecule")
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    # Calculate distances
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # Calculate clusters
    clusters = Butina.ClusterData(dists, nfps, config.cutoff, isDistData=True)
    clusters = sorted(clusters, key=lambda x: -len(x))
    cluster_counts = pd.Series([len(cluster) for cluster in clusters])

    # Make sure there are clusters above min_cluster_size
    if (cluster_counts < config.min_cluster_size).all():
        raise ValueError(
            f"All clusters are less than minimum cluster size {config.min_cluster_size}. Consider increasing cutoff to make clusters bigger."
        )
    # Combine small clusters
    elif (cluster_counts < config.min_cluster_size).any():
        # Identify the clusters smaller than the minimum size
        small_cluster_counts = cluster_counts[cluster_counts < config.min_cluster_size]

        for i in small_cluster_counts.index:
            # Find the smallest cluster bigger than the minimum size (receiving cluster)
            receiving_cluster_index = (
                cluster_counts[cluster_counts > config.min_cluster_size]
                .sort_values(ascending=True)
                .index[0]
            )

            # Add the small cluster to the receiving cluster identified above
            clusters[receiving_cluster_index] = set(
                list(clusters[receiving_cluster_index]) + list(clusters[i])
            )

            # Make the small cluster a null cluster
            clusters[i] = set()
        # Remove null clusters
        clusters = [cluster for cluster in clusters if len(cluster) != 0]

    # Add clusters to DataFrame
    df[config.cluster_column] = 0
    for i, cluster in enumerate(clusters):
        df.at[list(cluster), config.cluster_column] = i

    return clusters, df


@op(
    config_schema=dict(
        cluster_column=Field(str, description="Name of the column with the clusters"),
        log_scale=Field(
            bool, description="Whether the yscale should be in log", default_value=False
        ),
    ),
    out=Out(io_manager_key="reporting_mpl_io_manager"),
)
def plot_cluster_counts(context, df: pd.DataFrame):
    """Bar plot of cluster counts"""
    fig, ax = plt.subplots(1)
    df[context.solid_config["cluster_column"]].value_counts().plot.bar(ax=ax)
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Counts")
    if context.solid_config["log_scale"]:
        ax.set_yscale("log")
    ax.set_xticklabels([])
    return fig

def check_range(series, min_value, max_value,):
    return (series < max_value).all() and (series >= min_value).all()

@op(
    config_schema=dict(
        min_value=Field(float, description="Minimum value for output"),
        max_value=Field(float, description="Max value for output"),
    )
)
def limit_outputs(context, df: pd.DataFrame):
    # Limit outputs to minimum and maximum values
    config = RecursiveNamespace(**context.solid_config)
    groups = []
    for _, group in df.groupby(["smiles_1", "smiles_2"]):
        check = (
            check_range(group["ln_gamma_1"], config.min_value, config.max_value) and 
            check_range(group["ln_gamma_2"], config.min_value, config.max_value)
        )
        if check:
            groups.append(group)
    return pd.concat(groups, axis=0).reset_index()

@op(
    config_schema=dict(
        valid_size=Field(
            float,
            description="Fraction of data in validation set",
        ),
        test_size=Field(float, description="Fraction data in test set"),
    ),
    out=dict(train_inds=Out(), valid_inds=Out(), test_inds=Out()),
)
def cluster_split(context, clusters: list, data: pd.DataFrame):
    """This is mostly copied from DeepChem ButinaSplitter"""
    config = RecursiveNamespace(**context.solid_config)
    train_size = 1.0 - config.test_size - config.valid_size

    # Split according to clusters
    train_cutoff = train_size * len(data)
    valid_cutoff = (train_size + config.valid_size) * len(data)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    for cluster in clusters:
        if len(train_inds) + len(cluster) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(cluster) > valid_cutoff:
                test_inds += cluster
            else:
                valid_inds += cluster
        else:
            train_inds += cluster

    context.log.info(f"Unique molecules in train: {len(train_inds)}")
    context.log.info(f"Unique molecules in valid: {len(valid_inds)}")
    context.log.info(f"Unique molecules in test: {len(test_inds)}")
    return train_inds, valid_inds, test_inds


@op(
    config_schema=dict(
        # cluster_df_smiles_column=str,
        subsample_valid_cont=Field(float),
        smiles_columns=Field(
            [str],
            description="SMILES input columns",
            default_value=["smiles_1", "smiles_2"],
        ),
        target_columns=Field(
            [str],
            description="Target columns to be predicted by chemprop",
            default_value=["ln_gamma_1", "ln_gamma_2"],
        ),
        features_columns=Field(
            [str],
            description="List of extra features columns",
            default_value=["temperature (K)", "x(1)"],
        ),
    ),
    out=dict(
        train=Out(
            description="Indices of the training set",
            io_manager_key="model_input_csv_io_manager",
        ),
        train_features=Out(
            description="Indices of the training set",
            io_manager_key="model_input_csv_io_manager",
        ),
        valid_cont=Out(
            description="Validation containing molecules in the training set at temperatures and compositions not used during training.",
            io_manager_key="model_input_csv_io_manager",
        ),
        valid_cont_features=Out(
            description="Validation containing molecules in the training set at temperatures and compositions not used during training.",
            io_manager_key="model_input_csv_io_manager",
        ),
        valid_mix=Out(
            description="All combinations of training molecules and validation molecules",
            io_manager_key="model_input_csv_io_manager",
        ),
        valid_mix_features=Out(
            description="All combinations of training molecules and validation molecules",
            io_manager_key="model_input_csv_io_manager",
        ),
        valid_indp=Out(
            description="Just combinations of molecules in the validation set",
            io_manager_key="model_input_csv_io_manager",
        ),
        valid_indp_features=Out(
            description="Just combinations of molecules in the validation set",
            io_manager_key="model_input_csv_io_manager",
        ),
        test_mix=Out(
            description="All combinations of training molecules and test molecules",
            io_manager_key="model_input_csv_io_manager",
        ),
        test_mix_features=Out(
            description="All combinations of training molecules and test molecules",
            io_manager_key="model_input_csv_io_manager",
        ),
        test_indp=Out(
            description="Just combinations of molecules in the test set",
            io_manager_key="model_input_csv_io_manager",
        ),
        test_indp_features=Out(
            description="Just combinations of molecules in the test set",
            io_manager_key="model_input_csv_io_manager",
        ),
    ),
)
def merge_cluster_split(
    context,
    train_inds: list,
    valid_inds: list,
    test_inds: list,
    clusters_df: pd.DataFrame,
    data: pd.DataFrame,
):
    all_indices = {}
    config = RecursiveNamespace(**context.solid_config)
    # Create train, valid and test subset molecule_list dfs
    train_mol_df, valid_mol_df, test_mol_df = (
        clusters_df.iloc[train_inds],
        clusters_df.iloc[valid_inds],
        clusters_df.iloc[test_inds],
    )

    # Assign pairs which only contain molecules from the train set to train set
    train_base_indices = data[
        (data["smiles_1"].isin(train_mol_df["smiles"]))
        & (data["smiles_2"].isin(train_mol_df["smiles"]))
    ].index.to_numpy()

    # Subsample to exclude some temperatures / compositions for valid CONT
    n_train_base = len(train_base_indices)
    rng = np.random.default_rng(1995)
    train_select = rng.choice(
        n_train_base,
        size=int((1.0 - config.subsample_valid_cont) * n_train_base),
        replace=False,
    )
    mask = np.zeros(n_train_base, dtype=bool)
    mask[train_select] = True
    all_indices['train'] = train_base_indices[mask]
    all_indices['valid_cont'] = train_base_indices[~mask]
    context.log.info(f"Training set size: {len(all_indices['train'])}")
    context.log.info(f"Validation CONT size: {len(all_indices['valid_cont'] )}")

    # Validation MIX and INDP
    all_indices["valid_mix"] = data[
        (
            (data["smiles_1"].isin(train_mol_df["smiles"]))
            & (data["smiles_2"].isin(valid_mol_df["smiles"]))
        )
        | (
            (data["smiles_1"].isin(valid_mol_df["smiles"]))
            & (data["smiles_2"].isin(train_mol_df["smiles"]))
        )
    ].index.to_numpy()
    context.log.info(f"Validation MIX size: {len(all_indices['valid_mix'])}")
    all_indices["valid_indp"] = data[
        (data["smiles_1"].isin(valid_mol_df["smiles"]))
        & (data["smiles_1"].isin(valid_mol_df["smiles"]))
    ].index.to_numpy()
    context.log.info(f"Validation INDP size: {len(all_indices['valid_indp'] )}")

    # Test MIX and INDP
    all_indices["test_mix"] = data[
        (
            (data["smiles_1"].isin(train_mol_df["smiles"]))
            & (data["smiles_2"].isin(test_mol_df["smiles"]))
        )
        | (
            (data["smiles_1"].isin(test_mol_df["smiles"]))
            & (data["smiles_2"].isin(train_mol_df["smiles"]))
        )
    ].index.to_numpy()
    context.log.info(f"Test MIX size: {len(all_indices['test_mix'])}")
    all_indices["test_indp"] = data[
        (data["smiles_1"].isin(test_mol_df["smiles"]))
        & (data["smiles_1"].isin(test_mol_df["smiles"]))
    ].index.to_numpy()
    context.log.info(f"Test INDP size: {len(all_indices['test_indp'])}")

    for name, ind in all_indices.items():
        yield Output(
            data.iloc[ind][config.smiles_columns + config.target_columns],
            name
        )
        yield Output(
            data.iloc[ind][config.features_columns],
            f"{name}_features"
        )


@op(
    config_schema=dict(
        smiles_columns=Field(
            [str],
            description="SMILES input columns",
            default_value=["smiles_1", "smiles_2"],
        ),
        target_columns=Field(
            [str],
            description="Target columns to be predicted by chemprop",
            default_value=["ln_gamma_1", "ln_gamma_2"],
        ),
        features_columns=Field(
            [str],
            description="List of extra features columns",
            default_value=["temperature (K)", "x(1)"],
        ),
    ),
    out=dict(
        data_no_features=Out(pd.DataFrame, io_manager_key="model_input_csv_io_manager"),
        features=Out(pd.DataFrame, io_manager_key="model_input_csv_io_manager"),
    ),
)
def chemprop_split_molecules_features(context, data: pd.DataFrame):
    config = RecursiveNamespace(**context.solid_config)
    return (
        data[config.smiles_columns + config.target_columns],
        data[config.features_columns],
    )


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


@op(out=Out(io_manager_key="reporting_pil_io_manager"))
def visualize_scaffolds(
    df: pd.DataFrame, scaffolds: dict, selection_indices: np.ndarray = None
) -> Image:
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


@op(out=Out(io_manager_key="reporting_mpl_io_manager"))
def plot_scaffold_counts(df: pd.DataFrame):
    """Bar plot of scaffold frequency"""
    fig, ax = plt.subplots(1)
    df["scaffold_index"].value_counts().plot.bar(ax=ax)
    ax.set_xlabel("Scaffolds")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.set_xticklabels([])
    return fig
