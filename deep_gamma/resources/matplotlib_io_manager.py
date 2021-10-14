import os
from typing import Union

import pandas
from dagster import (
    AssetKey,
    EventMetadataEntry,
    IOManager,
    OutputContext,
    check,
    io_manager,
)
from matplotlib.figure import Figure


class MplIOManager(IOManager):
    """
    This IOManager will save a matplotlib figure as PNG

    Resource config
    base_path : The base path

    Config
    filename: should include the extension to save int
    dpi : Dots per inch to save figure.
    """

    def _get_path(self, context: OutputContext):
        base_path = context.resource_config["base_path"]
        filename = context.config["filename"]

        return os.path.join(base_path, filename)

    def get_output_asset_key(self, context: OutputContext):
        return AssetKey(
            [*context.resource_config["base_path"].split("://"), context.name]
        )

    def handle_output(
        self,
        context: OutputContext,
        obj: Figure,
    ):
        path = self._get_path(context)
        dpi = context.config["dpi"]
        if isinstance(obj, Figure):
            obj.savefig(path, dpi=dpi)
        else:
            raise Exception(f"Outputs of type {type(obj)} not supported.")
        yield EventMetadataEntry.path(path=path, label="path")

    def load_input(self, context):
        return super().load_input(context)


@io_manager(
    config_schema={"base_path": str},
    output_config_schema={"filename": str, "dpi": int},
)
def mpl_io_manager(_):
    return MplIOManager()
