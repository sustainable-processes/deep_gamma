import os

from dagster import (
    AssetKey,
    EventMetadataEntry,
    IOManager,
    OutputContext,
    Field,
    io_manager,
)
import numpy as np


class NpIOManager(IOManager):
    """
    This IOManager will save a matplotlib figure as PNG

    Resource config
    base_path : The base path

    Config
    filename: should include the extension to save int
    save_txt: If True saves as a text file
    """

    def _get_path(self, context: OutputContext):
        base_path = context.resource_config["base_path"]
        save_txt = context.resource_config.get("save_txt")
        compress = context.resource_config.get("compress")
        filename = context.config.get("filename")

        if filename:
            return os.path.join(base_path, filename)
        elif save_txt and compress:
            return os.path.join(base_path, f"{context.name}.txt.gz")
        elif save_txt and not compress:
            return os.path.join(base_path, f"{context.name}.txt")
        else:
            return os.path.join(base_path, f"{context.name}.npy")

    def get_output_asset_key(self, context: OutputContext):
        return AssetKey(
            [*context.resource_config["base_path"].split("://"), context.name]
        )

    def handle_output(
        self,
        context: OutputContext,
        obj: np.ndarray,
    ):
        path = self._get_path(context)
        save_txt = context.resource_config.get("save_txt")
        if isinstance(obj, np.ndarray):
            if save_txt:
                np.savetxt(path, obj)
            else:
                np.save(path, obj)
        else:
            raise Exception(f"Outputs of type {type(obj)} not supported.")
        yield EventMetadataEntry.path(path=path, label="path")

    def load_input(self, context):
        return super().load_input(context)


@io_manager(
    config_schema={
        "base_path": str,
        "save_txt": Field(
            bool, default_value=False, description="Save as text instead of binary."
        ),
        "compress": Field(
            bool,
            default_value=False,
            description="Whether to compress the file if save_txt is True.",
        ),
    },
    output_config_schema={
        "filename": Field(
            str, is_required=False, description="Override the default filename"
        ),
    },
)
def np_io_manager(_):
    return NpIOManager()
