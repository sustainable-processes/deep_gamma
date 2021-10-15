from chemprop.train import run_training, cross_validate
from chemprop.args import TrainArgs
import wandb

from pathlib import Path


class VLETrainArgs(TrainArgs):
    experiment_name: str = "cosmo"
    lr_scheduler: str = "Noam"


def train_model():
    # Get args
    args = VLETrainArgs().parse_args()

    # Setup wandb
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
