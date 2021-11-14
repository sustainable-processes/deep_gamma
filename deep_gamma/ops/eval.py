from unittest import result
from chemprop.models import model
from chemprop.train import make_predictions
from chemprop.args import PredictArgs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from pathlib import Path
from typing import List, Optional
import wandb
import os
from chemprop.args import CommonArgs


def parity_plot(df: pd.DataFrame, target_columns: List[str]):
    fig, axes = plt.subplots(1, len(target_columns), figsize=(5*len(target_columns), 5))
    if type(axes) != np.ndarray:
        axes = [axes]
    fig.subplots_adjust(wspace=0.2)
    c = "#025b66"
    for i, target_column in enumerate(target_columns):
        # Parity plot
        axes[i - 1].scatter(
            df[target_column], df[f"{target_column}_pred"], alpha=0.01, c=c
        )
        axes[i - 1].set_xlabel(f"Measured {target_column}")
        axes[i - 1].set_ylabel(f"Predicted {target_column}")
        max_val = df[target_column].max()
        min_val = df[target_column].min()

        # Parity line
        axes[i - 1].plot([min_val, max_val], [min_val, max_val], "--", c="grey")

        # Scores
        rmse = mean_squared_error(df[target_column], df[f"{target_column}_pred"]) ** (
            0.5
        )
        mae = mean_absolute_error(df[target_column], df[f"{target_column}_pred"])
        rmse_patch = mpatches.Patch(label="RMSE = {:.3f}".format(rmse), color=c)
        mae_patch = mpatches.Patch(label="MAE = {:.3f}".format(mae), color=c)
        axes[i - 1].legend(handles=[rmse_patch, mae_patch])

        # Title
        axes[i - 1].set_title(target_column, fontsize=16)
    return fig, axes

def absolute_error_composition(df: pd.DataFrame):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    fig.subplots_adjust(wspace=0.2)
    big_df_errors = df.dropna().copy().reset_index()
    for i in [1,2]:
        abs_difference = (df.dropna()[f"ln_gamma_{i}"]-df.dropna()[f"ln_gamma_{i}_pred"]).abs()
        big_df_errors[f"abs_error_{i}"]= abs_difference.to_numpy()
        axes[i-1].scatter(df.dropna()["x(1)"], abs_difference, alpha=0.1, c = "#025b66")
        axes[i-1].set_xlabel("x(1)")
        axes[i-1].set_ylabel(f"Absolute Error $\ln\gamma_{i}$")
        axes[i-1].set_title(f"$\ln\gamma_{i}$", fontsize=16)
    return fig, axes

def calculate_activity_coefficients_polynomial(preds):
    pass
    
class VLEPredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""
    data_dir: Optional[str] = "data/"
    skip_prediction: bool = False
    smiles_columns: List[str] = ["smiles_1", "smiles_2"]
    number_of_molecules: int = 2
    drop_na: bool = False
    drop_extra_columns: bool = False
    """Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns."""
    ensemble_variance: bool = False
    """Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path."""
    individual_ensemble_predictions: bool = False
    """Whether to return the predictions made by each of the individual models rather than the average of the ensemble"""
    polynomial: bool = False
    num_workers: int = 4

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        self.data_input_dir = Path(self.data_dir) / "05_model_input"
        results_path = Path("results/")
        os.makedirs(results_path, exist_ok=True)
        self.output_path = results_path / "07_model_output"
        os.makedirs(self.output_path, exist_ok=True)
        self.reporting_dir = results_path / "08_reporting"
        os.makedirs(self.reporting_dir, exist_ok=True)
        super().process_args( )



def evaluate():
    # Arguments
    args = VLEPredictArgs().parse_args()

    # Download models
    model_paths = {}
    model_run_ids = {
        "cosmo_base": "zn669uuj",
        "cosmo_base_pretrained": "1tsddx25",
        # "cosmo_polynomial_pretrained": "3isfpnw2",
        # "cosmo_polynomial": "3nd8gspj"
    }
    wandb.login(key="eddd91debd4aeb24f212695d6c663f504fdb7e3c")
    for name, run_id in model_run_ids.items():
        if "polynomial" in name:
            path = "fold_0/model_0/model.pt" 
        else:
            path = "model_0/model.pt"
        wandb_base_path = f"ceb-sre/vle/{run_id}"
        checkpoint_path = wandb.restore(path, run_path=wandb_base_path)
        model_paths[name] = str(checkpoint_path.name)

    # Loop through different validation and test sets.
    sets = ["valid_cont", "valid_mix", "valid_indp", "test_indp", "test_mix"]
    for model_name, model_path in model_paths.items():
        args.checkpoint_paths = [model_path]
        for predict_set in sets:
            # Paths
            args.test_path = args.data_input_dir /  f"{predict_set}.csv"
            if "polynomial" in model_name:
                args.features_path = [args.data_input_dir / f"{predict_set}_features.csv"]
            else:
                args.features_path = [args.data_input_dir / f"{predict_set}_features.csv"]
            args.preds_path = args.output_path / f"{predict_set}_preds.csv"
 
            # Make predictions
            if not args.skip_prediction:
                # Make predictions
                preds = make_predictions(args)

            if "polynomial" in model_name:
                calculate_activity_coefficients_polynomial(preds)

            # Read back in test predictions data
            preds = pd.read_csv(args.preds_path)
            preds = preds.rename(columns=lambda t: f"{t}_pred")
            truth = pd.read_csv(args.test_path)
            if len(args.features_path) > 0:
                features = pd.read_csv(args.features_path[0])
                big_df = pd.concat([truth, features, preds], axis=1)
            else:
                big_df = pd.concat([truth, preds], axis=1)

            # Parity plot
            if args.drop_na:
                big_df = big_df.dropna()
            fig, _ = parity_plot(big_df, args.target_columns)
            fig.savefig(args.reporting_dir / f"{model_name}_{predict_set}_parity_plot.png", dpi=300)

            # Absolute error vs composition
            fig, _ = absolute_error_composition(big_df)
            fig.savefig(args.reporting_dir / f"{predict_set}_absolute_error_vs_composition.png", dpi=300)


if __name__ == "__main__":
    evaluate()