import wandb

import os
import numpy as np
from pathlib import Path
from typing import Optional, List
from typing_extensions import Literal


from collections import defaultdict
import csv
import json
from logging import Logger
import os
from typing import Callable, Dict, List, Tuple
from functools import partialmethod
import numpy as np
import pandas as pd
from pathlib import Path
import json

from chemprop.args import TrainArgs, Metric
from chemprop.train.cross_validate import cross_validate
from chemprop.train.run_training import run_training



class VLETrainArgs(TrainArgs):
    experiment_name: str
    artifact_name: str
    data_dir: Optional[str] = "data/"
    data_path: Optional[str] = None
    lr_scheduler: str = "Noam"
    split_type: Literal[
        "random",
        "scaffold_balanced",
        "predetermined",
        "crossval",
        "cv",
        "cv-no-test",
        "index_predetermined",
        "random_with_repeated_smiles",
        "custom",
    ] = "custom"
    """Method of splitting the data into train/val/test."""
    number_of_molecules: int = 2
    dataset_type: Literal["regression", "classification", "multiclass"] = "regression"
    smiles_columns: List[str] = ["smiles_1", "smiles_2"]
    target_columns: List[str] = ["ln_gamma_1", "ln_gamma_2"]
    epochs: int = 100
    num_workers: int = 7
    cache_cutoff: int = int(1e9)
    save_preds: bool = True
    extra_metrics: List[str] = ["r2", "mae"]
    metric: Metric = "mse"
    mpn_shared: bool = True
    depth: int = 4
    hidden_size: int = 200
    activation: str = "LeakyReLU"
    

    def process_args(self) -> None:
        data_dir = Path(self.data_dir) / "05_model_input"
        if self.data_path is None:
            self.data_path = str(data_dir / "data_no_features.csv")
        if self.features_path is None:
            self.features_path = [str(data_dir / "features.csv")]

        super().process_args()

        if self.split_type == "custom":
            train_indices = np.loadtxt(data_dir / "train_indices.txt").tolist()
            valid_indices = np.loadtxt(data_dir / "valid_mix_indices.txt").tolist()
            test_indices = []

            self._crossval_index_sets = [[train_indices, valid_indices, test_indices]]
            self.split_type = "index_predetermined"


def train_model():
    # Get args
    args = VLETrainArgs().parse_args()

    # Setup wandb
    wandb.login(key="eddd91debd4aeb24f212695d6c663f504fdb7e3c")
    run = wandb.init(entity="ceb-sre", project="vle", name=args.experiment_name)
    wandb.tensorboard.patch(save=False, tensorboardX=True, pytorch=True)
    wandb.config.update(args.as_dict())

    # Change save_dir to wandb run directory
    args.save_dir = wandb.run.dir
    save_dir = Path(args.save_dir)

    # Save files to cloud as the run progresses
    files_to_save = [
        save_dir / "fold_0" / "*.csv",
        save_dir / "args.json",
        save_dir / "fold_0/model_0/model.pt",
    ]
    for file in files_to_save:
        wandb.save(str(file), base_path=str(save_dir))

    # Run training
    cross_validate(args=args, train_func=run_training)

    # Save model as an artifact
    artifact = wandb.Artifact(args.artifact_name, type="model")
    artifact.add_file(save_dir / "fold_0/model_0/model.pt")
    run.log_artifact(artifact)


if __name__ == "__main__":
    train_model()
