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
from PIL.Image import Image


class PilIOManager(IOManager):
    """
    This IOManager will save a PIL image as PNG
    """

    def _get_path(self, context: OutputContext):
        base_path = context.resource_config["base_path"]
        filename = context.config["filename"].rstrip(".png")

        return os.path.join(base_path, f"{filename}.png")

    def get_output_asset_key(self, context: OutputContext):
        return AssetKey(
            [*context.resource_config["base_path"].split("://"), context.name]
        )

    def handle_output(
        self,
        context: OutputContext,
        obj: Image,
    ):
        path = self._get_path(context)
        if isinstance(obj, Image):
            obj.save(path)
        else:
            raise Exception(f"Outputs of type {type(obj)} not supported.")
        yield EventMetadataEntry.path(path=path, label="path")

    def load_input(self, context):
        return super().load_input(context)


@io_manager(config_schema={"base_path": str}, output_config_schema={"filename": str})
def pil_io_manager(_):
    return PilIOManager()
