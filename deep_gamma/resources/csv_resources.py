import pandas as pd
from dagster import (
    root_input_manager,
)


@root_input_manager(config_schema={"path": str})
def csv_loader(context):
    return pd.read_csv(context.resource_config["path"])
