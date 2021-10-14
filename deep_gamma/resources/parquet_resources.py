import os
from typing import Union

import pandas as pd
from dagster import (
    AssetKey,
    EventMetadataEntry,
    IOManager,
    OutputContext,
    check,
    io_manager,
    root_input_manager,
)


@root_input_manager(config_schema={"path": str})
def parquet_loader(context):
    return pd.read_parquet(context.resource_config["path"])


class ParquetIOManager(IOManager):
    """
    This IOManager will take in a pd or pyspark dataframe and store it in parquet at the
    specified path.
    Downstream ops can either load this dataframe into a spark session or simply retrieve a path
    to where the data is stored.
    """

    def _get_path(self, context: OutputContext):

        base_path = context.resource_config["base_path"]

        return os.path.join(base_path, f"{context.name}.pq")

    def get_output_asset_key(self, context: OutputContext):
        return AssetKey(
            [*context.resource_config["base_path"].split("://"), context.name]
        )

    def handle_output(
        self,
        context: OutputContext,
        obj: pd.DataFrame,
    ):

        path = self._get_path(context)
        if isinstance(obj, pd.DataFrame):
            row_count = len(obj)
            obj.to_parquet(path=path, index=False)
        else:
            raise Exception(f"Outputs of type {type(obj)} not supported.")
        yield EventMetadataEntry.int(value=row_count, label="row_count")
        yield EventMetadataEntry.path(path=path, label="path")

    def load_input(self, context) -> Union[pd.DataFrame, str]:
        # In this load_input function, we vary the behavior based on the type of the downstream input
        path = self._get_path(context.upstream_output)
        if context.dagster_type.typing_type == pd.DataFrame:
            # return pyspark dataframe
            return pd.read_parquet(path)
        elif context.dagster_type.typing_type == str:
            # return path to parquet files
            return path
        return check.failed(
            f"Inputs of type {context.dagster_type} not supported. Please specify a valid type "
            "for this input either in the solid signature or on the corresponding InputDefinition."
        )


@io_manager(
    config_schema={"base_path": str},
)
def parquet_io_manager(_):
    return ParquetIOManager()
