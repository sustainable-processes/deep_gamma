from chemprop.train import run_training, cross_validate
from chemprop.args import TrainArgs
import wandb

import os
import numpy as np
from pathlib import Path
from typing import Optional, List
from typing_extensions import Literal


class VLETrainArgs(TrainArgs):
    data_dir: Optional[str] = None
    data_path: Optional[str] = None
    experiment_name: str = "cosmo"
    lr_scheduler: str = "Noam"
    split_type = "cv-no-test"
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
    save_preds: bool = True
    extra_metrics: List[str] = ["r2", "mae"]
    mpn_shared: bool = True

    def process_args(self) -> None:
        data_dir = Path(self.data_dir) / "05_model_input"
        if self.data_path is None:
            self.data_path = str(data_dir / "data_no_features.csv")
        if self.features_path is None:
            self.features_path = [str(data_dir / "features.csv")]

        super().process_args()

        if self.split_type == "custom":
            train_indices = np.loadtxt(data_dir / "train_indices.txt")
            valid_indices = np.loadtxt(data_dir / "valid_mix_indices.txt")
            test_indices = []

            self._crossval_index_sets = [[train_indices, valid_indices, test_indices]]
            self.split_type = "index_predetermined"


def train_model():
    # Get args
    args = VLETrainArgs().parse_args()

    # Setup wandb
    wandb.login(key="eddd91debd4aeb24f212695d6c663f504fdb7e3c")
    wandb.init(entity="ceb-sre", project="vle", name=args.experiment_name)
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


if __name__ == "__main__":
    train_model()
