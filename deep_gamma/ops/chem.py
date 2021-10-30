import wandb
from chemprop.args import TrainArgs, Metric
from chemprop.train.cross_validate import cross_validate
from chemprop.train.run_training import run_training

from typing import Optional, List
from typing_extensions import Literal
import os
from pathlib import Path



class VLETrainArgs(TrainArgs):
    experiment_name: str
    artifact_name: str
    data_dir: Optional[str] = "data/"
    data_path: Optional[str] = None
    lr_scheduler: str = "Noam"
    number_of_molecules: int = 2
    dataset_type: Literal["regression", "classification", "multiclass"] = "regression"
    smiles_columns: List[str] = ["smiles_1", "smiles_2"]
    target_columns: List[str] = ["ln_gamma_1", "ln_gamma_2"]
    epochs: int = 100
    num_workers: int = 3
    cache_cutoff: int = int(1e9)
    save_preds: bool = True
    extra_metrics: List[str] = ["r2", "mae"]
    metric: Metric = "mse"
    mpn_shared: bool = True
    depth: int = 4
    hidden_size: int = 200
    activation: str = "LeakyReLU"
    batch_size: int = 1000
    wandb_checkpoint_run: str = None
    wandb_entity: str = "ceb-sre"
    wandb_project: str = "vle"
    use_molecule_weights: bool = False
    combisolv: bool = False
    polynomial: bool = False

    def process_args(self) -> None:
        if not self.combisolv and not self.polynomial:
            data_dir = Path(self.data_dir) / "05_model_input"
            # Train
            if self.data_path is None:
                self.data_path = str(data_dir / "train.csv")
            if self.features_path is None and not self.use_molecule_weights:
                self.features_path = [str(data_dir / "train_features.csv")]
            elif self.features_path is None and  self.use_molecule_weights:
                self.features_path = [str(data_dir / "train_temperatures.csv")]
            if self.use_molecule_weights and self.molecule_weights_path is None:
                self.molecule_weights_path = str(data_dir / "train_weights.csv")

            # Validation
            if self.separate_val_path is None:
                self.separate_val_path = str(data_dir / "valid_mix.csv")
            if self.separate_val_features_path is None and not self.use_molecule_weights:
                self.separate_val_features_path = [str(data_dir / "valid_mix_features.csv")]
            elif self.separate_val_features_path is None and self.use_molecule_weights:
                self.separate_val_features_path = [str(data_dir / "valid_mix_temperatures.csv")]
            if self.use_molecule_weights and self.separate_val_molecule_weights_path is None:
                self.separate_val_molecule_weights_path = str(data_dir / "valid_mix_weights.csv")

            # Test
            if self.separate_test_path is None:
                self.separate_test_path = str(data_dir / "test_mix.csv")
            if self.separate_test_features_path is None and not self.use_molecule_weights:
                self.separate_test_features_path = [str(data_dir / "test_mix_features.csv")]
            elif self.separate_test_features_path is None and  self.use_molecule_weights:
                self.separate_test_features_path = [str(data_dir / "test_mix_temperatures.csv")]
            if self.use_molecule_weights and self.separate_test_molecule_weights_path is None:
                self.separate_test_molecule_weights_path = str(data_dir / "test_mix_weights.csv")
        elif self.polynomial:
            self.target_columns = ['c0_0', 'c0_1', 'c1_0', 'c1_1', 'c2_0', 'c2_1', 'c3_0', 'c3_1', 'c4_0', 'c4_1']
        else:
            data_dir = Path(self.data_dir)
            self.data_path = data_dir / "combisolv.txt"
            self.smiles_columns = ["mol solvent", "mol solute"]
            self.target_columns = ["target Gsolv kcal"]
            self.max_lr = 0.0002
            self.init_lr = 0.0001
            self.ffn_hidden_size = 500

        super().process_args()

        # if self.split_type == "custom":
        #     train_indices = np.loadtxt(data_dir / "train_indices.txt").astype(int)
        #     valid_indices = np.loadtxt(data_dir / "valid_mix_indices.txt").astype(int)
        #     test_indices = []

        #     self._crossval_index_sets = [[train_indices, valid_indices, test_indices]]
        #     self.split_type = "index_predetermined"

def get_grid_info()->dict:
    """Information about Grid.ai run if available"""
    d = {}
    d["grid_experiment_name"] = os.environ.get("GRID_EXPERIMENT_NAME")
    d["grid_experiment_id"] = os.environ.get("GRID_EXPERIMENT_ID")
    d["grid_instance_type"] = os.environ.get("GRID_INSTANCE_TYPE")
    d["grid_user"] = os.environ.get("GRID_USER_ID")
    d["grid_cluster_id"] = os.environ.get("GRID_CLUSTER_ID")
    return d

def train_model():
    # Get args
    args = VLETrainArgs().parse_args()

    # Setup wandb
    wandb.login(key="eddd91debd4aeb24f212695d6c663f504fdb7e3c")
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.experiment_name)
    wandb.tensorboard.patch(save=False, tensorboardX=True, pytorch=True)
    # Don't put all the split data on wandb
    d = args.as_dict()
    d.pop("crossval_index_sets")
    # Add grid information if available
    d.update(get_grid_info())
    wandb.config.update(d)

    # Download checkpoint model if specified
    if args.wandb_checkpoint_run is not None and args.checkpoint_dir is None:
        wandb_base_path = f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_checkpoint_run}"
        checkpoint_path = wandb.restore("fold_0/model_0/model.pt", run_path=wandb_base_path)
        args.checkpoint_path = str(checkpoint_path.name)
    elif args.wandb_checkpoint_run is not None and args.checkpoint_dir is not None:
        ValueError("Can only have one of the following: wandb_checkpoint_run and checkpoint_dir")


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
