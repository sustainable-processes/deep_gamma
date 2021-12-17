"""Loads a trained chemprop model checkpoint and encode latent fingerprint vectors for the molecules in a dataset.
    Uses the same command line arguments as predict."""

from eval import VLEPredictArgs
from chemprop.train.molecule_fingerprint import molecule_fingerprint
from chemprop.args import FingerprintArgs
import os
from typing import List, Optional
from typing_extensions import Literal
from pathlib import Path
import wandb

class DeepGammaFingerprintArgs(VLEPredictArgs):
    wandb_checkpoint_run: Optional[str] = None
    wandb_entity: str = "ceb-sre"
    wandb_project: str = "vle"
    fingerprint_type: Literal['MPN','last_FFN'] = 'MPN'
    def process_args(self) -> None:
        wandb.login(key="eddd91debd4aeb24f212695d6c663f504fdb7e3c")
        if self.wandb_checkpoint_run is not None and self.checkpoint_dir is None:
            wandb_base_path = f"{self.wandb_entity}/{self.wandb_project}/{self.wandb_checkpoint_run}"
            try:
                checkpoint_path = wandb.restore("fold_0/model_0/model.pt", run_path=wandb_base_path)
            except ValueError:
                checkpoint_path = wandb.restore("model_0/model.pt", run_path=wandb_base_path)
            self.checkpoint_path = str(checkpoint_path.name)
        elif self.wandb_checkpoint_run is not None and self.checkpoint_dir is not None:
            ValueError("Can only have one of the following: wandb_checkpoint_run and checkpoint_dir")
        super().process_args()

def deep_gamma_fingerprint() -> None:
    """
    Parses Chemprop predicting arguments and returns the latent representation vectors for
    provided molecules, according to a previously trained model.
    """
    args = DeepGammaFingerprintArgs().parse_args()
    sets = ["train", "valid_cont", "valid_mix", "valid_indp", "test_indp", "test_mix"]
    for predict_set in sets:
        args.test_path = args.data_input_dir /  f"{predict_set}.csv"
        args.features_path = [args.data_input_dir / f"{predict_set}_features.csv"]
        args.preds_path = args.output_path / f"{predict_set}_preds.csv"
        molecule_fingerprint(args=args)


if __name__ == '__main__':
    deep_gamma_fingerprint()