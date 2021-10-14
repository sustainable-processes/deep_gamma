"""
Resolve molecules in any representation to SMILES strings that can be read by chemprop.

"""
from ast import For
from numpy import insert
import pandas as pd
import cirpy
from tqdm.auto import tqdm
from dagster import op, In, Out, Field
from deep_gamma import RecursiveNamespace


@op(
    ins=dict(data=In(root_manager_key="molecule_list_loader")),
    out=dict(
        molecule_list_with_smiles=Out(io_manager_key="intermediate_parquet_io_manager")
    ),
    config_schema=dict(
        input_column=Field(
            str, description="The name column with strings to be resolved to SMILES"
        ),
        smiles_column=Field(
            str,
            description="""The name of the created SMILES column. Defaults to "smiles". """,
        ),
    ),
)
def lookup_smiles(
    context,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve molecules in any representation to SMILES strings

    The idea is that you have a list of all the unique molecules
    from the data generation step, and you use that generate
    the SMILES strings which can be merged with the generated data
    using the `resolve_smiles` function.

    Notes
    -----
    This uses cirpy to do resolution: https://cirpy.readthedocs.io/
    Skips lookup for any rows where `smiles_column` has already been resolved.

    """
    config = RecursiveNamespace(**context.solid_config)

    # Check if the smiles column already exists
    if config.smiles_column in data.columns:
        any_missing = data[config.smiles_column].isna().any()
        if not any_missing:
            return data
    else:
        data[config.smiles_column] = ""

    # Resolve
    smiles = [
        cirpy.resolve(row[config.input_column], "smiles")
        for _, row in tqdm(data.iterrows(), total=data.shape[0])
        if row["smiles"] == ""
    ]
    data[config.smiles_column] = smiles

    return data


@op(
    ins=dict(df=In(root_manager_key="gamma_data_loader")),
    out=dict(data_with_smiles=Out(io_manager_key="intermediate_parquet_io_manager")),
    config_schema=dict(
        input_column_prefix=Field(str, description="Merge column prefix for df"),
        smiles_column_prefix=Field(str, description="Prefix for smiles columns in df"),
        molecule_df_input_column=Field(
            str, description="Merge column in molecule_list_df"
        ),
        molecule_df_smiles_column=Field(
            str, description="Column for merging SMILES in molecule_df"
        ),
        lookup_failed_smiles_as_name=Field(
            bool,
            description="Lookup cases where names were used instead of CAS numbers as names using the molecule_list_df_name_column",
        ),
        molecule_list_df_name_column=Field(
            str, "The name of the name column molecule_list_df"
        ),
    ),
)
def resolve_smiles(
    context,
    df: pd.DataFrame,
    molecule_list_df: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve a column in df with another df containing SMILES"""
    # Config
    config = RecursiveNamespace(**context.solid_config)

    # Copy datafrmae
    new_df = df.copy()

    # Drop unncessary columns to make merge faster
    drop_columns = molecule_list_df.columns.tolist()
    drop_columns.remove(config.molecule_df_smiles_column)
    drop_columns.remove(config.molecule_df_input_column)
    if config.molecule_list_df_name_column is not None:
        drop_columns.remove(config.molecule_list_df_name_column)
    molecule_list_df = molecule_list_df.drop(drop_columns, axis=1)

    # Do the merge
    for i in [1, 2]:
        new_df = pd.merge(
            new_df,
            molecule_list_df,
            left_on=f"{config.input_column_prefix}_{i}",
            right_on=config.molecule_df_input_column,
            how="left",
        )
        new_df = new_df.rename(
            columns={
                config.molecule_df_smiles_column: f"{config.smiles_column_prefix}_{i}"
            }
        ).drop(config.molecule_df_input_column, axis=1)

        # Deal with outputs where the COSMO name was used in place
        # of the CAS number
        if config.lookup_failed_smiles_as_name:
            # Mask out CAS numbers which are actually CAS numbers
            # So you just get ones that are names
            names = (
                df.mask(
                    df[f"{config.input_column_prefix}_{i}"].str.contains(
                        r"\b[1-9]{1}[0-9]{1,5}-\d{2}-\d\b", regex=True
                    )
                )
                .reset_index()
                .dropna()
                .drop("index", axis=1)
                .reset_index()
            )
            smiles_names = (
                pd.merge(
                    names,
                    molecule_list_df,
                    left_on=f"{config.input_column_prefix}_{i}",
                    right_on=config.molecule_list_df_name_column,
                )
                .set_index("index")
                .drop(config.molecule_list_df_name_column, axis=1)
            )
            smiles_names = smiles_names[[config.molecule_df_smiles_column]].rename(
                columns={
                    config.molecule_df_smiles_column: f"{config.smiles_column_prefix}_{i}"
                }
            )
            new_df.update(smiles_names)

        # Drop rows that failed to merge
        new_df = new_df.dropna()

    context.log.info(f"Total Rows Resolved: {new_df.shape[0]}")
    return new_df
