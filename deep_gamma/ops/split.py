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

from deep_gamma import RecursiveNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from umap import UMAP
import logging
from multiprocessing import Pool
from typing import Callable, List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray  # type: ignore
from tqdm import tqdm

from dagster import op, Out, Field, Output
from typing import Any, Optional, Tuple, Dict

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048
logger = logging.getLogger(__name__)


@op(
    config_schema=dict(
        smiles_column=Field(
            str,
            description="Column containing SMILES strings to use for forming clusters",
            default_value="smiles",
        ),
        cluster_column=Field(
            str,
            description="Name of the column to create with the clusters",
            default_value="cluster",
        ),
        drop_duplicates=Field(
            bool,
            description="Whether to drop duplicate SMILES strings before clustering",
            default_value=True,
        )
    ),
    out=dict(
        clusters=Out(),
        molecule_list_with_clusters=Out(
            io_manager_key="intermediate_parquet_io_manager"
        ),
    ),
)
def find_clusters(context, data: pd.DataFrame) -> Tuple[List, pd.DataFrame]:
    config = RecursiveNamespace(**context.solid_config)
    data = data.copy().reset_index(drop=True)
    kmeans_args: Optional[Dict[str, Any]] = None
    umap_before_cluster: bool = True
    umap_kwargs: Optional[Dict[str, Any]] = None

    if config.drop_duplicates:
        data = data.drop_duplicates(subset=config.smiles_column).reset_index()

    # Calculate fingerprints
    context.log.info("Calculating fingerprints...")
    fps = compute_morgan_fingerprints(data[config.smiles_column])

    # K-means clustering
    context.log.info("Clustering data...")
    kmeans_kwargs = kmeans_args if kmeans_args is not None else {"random_state": 0}
    if umap_before_cluster:
        if umap_kwargs is None:
            umap_kwargs = {}
        if "n_components" not in umap_kwargs:
            umap_kwargs["n_components"] = 5
        if "n_neighbors" not in umap_kwargs:
            umap_kwargs["n_neighbors"] = 15
        if "min_dist" not in umap_kwargs:
            umap_kwargs["min_dist"] = 0.1
        if "metric" not in umap_kwargs:
            umap_kwargs["metric"] = "jaccard"
        if "random_state" not in umap_kwargs:
            umap_kwargs["random_state"] = 0
        reducer = UMAP(**umap_kwargs)
        X: np.ndarray = reducer.fit_transform(fps)  # type: ignore
        num_nan = np.isnan(X).sum()
        if num_nan > 0:
            raise ValueError("UMAP returned NaN values.")
    else:
        X = fps
    kmeans = KMeans(**kmeans_kwargs)
    kmeans.fit(X)

    # Cluster labels
    cluster_labels = kmeans.labels_ + 1  # type: ignore
    data[config.cluster_column] = cluster_labels

    # Clusters
    clusters = []
    for cluster in range(1, cluster_labels.max() + 1):
        clusters.append(data[data[config.cluster_column] == cluster].index.tolist())

    return clusters, data


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
        metadata_columns=Field(
            [str],
            description="List of extra metadata columns to keep",
            default_value=["cas_number_1", "cas_number_2", "names_1", "names_2"],
        )
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
    train_indx: list,
    valid_indx: list,
    test_indx: list,
    clusters_df: pd.DataFrame,
    data: pd.DataFrame,
):
    all_indices = {}
    config = RecursiveNamespace(**context.solid_config)

    # Create train, valid and test subset molecule_list dfs
    train_mol_df, valid_mol_df, test_mol_df = (
        clusters_df.iloc[train_indx],
        clusters_df.iloc[valid_indx],
        clusters_df.iloc[test_indx],
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
    all_indices["train"] = train_base_indices[mask]
    all_indices["valid_cont"] = train_base_indices[~mask]
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
        & (data["smiles_2"].isin(test_mol_df["smiles"]))
    ].index.to_numpy()
    context.log.info(f"Test INDP size: {len(all_indices['test_indp'])}")

    for name, ind in all_indices.items():
        yield Output(
            data.iloc[ind][config.smiles_columns + config.target_columns + config.metadata_columns], name
        )
        yield Output(data.iloc[ind][config.features_columns], f"{name}_features")


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


def check_range(
    series,
    min_value,
    max_value,
):
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
    for _, group in df.groupby(["cas_number_1", "cas_number_2"]):
        check = check_range(
            group["ln_gamma_1"], config.min_value, config.max_value
        ) and check_range(group["ln_gamma_2"], config.min_value, config.max_value)
        if check:
            groups.append(group)
    return pd.concat(groups, axis=0).reset_index()


def _canonicalize_smiles(smi) -> Union[str, None]:
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None


def compute_morgan_fingerprint(
    mol: Union[str, Chem.Mol],  # type: ignore
    radius: int = MORGAN_RADIUS,
    num_bits: int = MORGAN_NUM_BITS,
) -> Union[np.ndarray, None]:
    """Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D boolean numpy array (num_bits,) containing the binary Morgan fingerprint.
    """
    try:
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol  # type: ignore
        morgan_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)  # type: ignore
        morgan_fp = np.zeros((1,))
        ConvertToNumpyArray(morgan_vec, morgan_fp)
        morgan_fp = morgan_fp.astype(bool)
    except Exception as e:
        logger.warning(f"Could not compute Morgan fingerprint for molecule: {mol}.")
        morgan_fp = np.zeros((num_bits,), dtype=bool)

    return morgan_fp


def compute_morgan_fingerprints(
    mols: List[str],
    radius: int = MORGAN_RADIUS,
    num_bits: int = MORGAN_NUM_BITS,
) -> np.ndarray:
    """Generates molecular fingerprints for each molecule in a list of molecules (in parallel).

    :param mols: A list of molecules (i.e., either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 2D numpy array (num_molecules, num_features) containing the fingerprints for each molecule.
    """
    return np.array(
        [
            compute_morgan_fingerprint(mol, radius=radius, num_bits=num_bits)
            for mol in tqdm(
                mols,
                total=len(mols),
                desc=f"Generating morgan fingerprints",
            )
        ],
        dtype=float,
    )
