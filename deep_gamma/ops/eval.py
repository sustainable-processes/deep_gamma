from unittest import result
from chemprop.models import model
from chemprop.train import make_predictions
from chemprop.args import PredictArgs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from pathlib import Path
from typing import List, Optional
from typing_extensions import Literal
import os
from chemprop.args import CommonArgs
import json
import wandb
import logging

def setup_logger(log_filename: str = "evaluation.log"):
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Std out handler
    std_handler = logging.StreamHandler()
    std_handler.setLevel(level=logging.ERROR)

    # create a file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level=logging.INFO)

    # create a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(std_handler)
    return logger

def parity_plot(
    df: pd.DataFrame,
    target_columns: List[str],
    format_gammas: bool = False,
    scores: dict = None,
    alpha=0.01,
):
    fig, axes = plt.subplots(1, len(target_columns), figsize=(5*len(target_columns), 5))
    if type(axes) != np.ndarray:
        axes = [axes]
    fig.subplots_adjust(wspace=0.4)
    c = "#025b66"
    axis_fontsize = 16
    for i, target_column in enumerate(target_columns):
        # Parity plot
        axes[i].scatter(
            df[target_column], df[f"{target_column}_pred"], alpha=alpha, c=c
        )
        if not format_gammas:
            axes[i].set_xlabel(f"Measured {target_column}", fontsize=axis_fontsize)
            axes[i].set_ylabel(f"Predicted {target_column}", fontsize=axis_fontsize)
        else:
            axes[i].set_xlabel(f"Measured $\ln \gamma_{i+1}$", fontsize=axis_fontsize)
            axes[i].set_ylabel(f"Predicted $\ln \gamma_{i+1}$", fontsize=axis_fontsize)
        axes[i].tick_params(direction="in", which="both", labelsize=12)
        max_val = df[target_column].max()
        min_val = df[target_column].min()

        # Parity line
        axes[i].plot([min_val, max_val], [min_val, max_val], "--", c="grey")

        # Scores
        if scores is not None:
            for i, target_column in enumerate(target_columns):
                rmse_patch = mpatches.Patch(label="RMSE = {:.3f}".format(scores[f"{target_column}_rmse"]), color=c)
                mae_patch = mpatches.Patch(label="MAE = {:.3f}".format(scores[f"{target_column}_mae"]), color=c)
                r2_score =  mpatches.Patch(label=r"$R^2$"+ "= {:.3f}".format(scores[f"{target_column}_r2"]), color=c)
                axes[i - 1].legend(handles=[rmse_patch, mae_patch, r2_score])


    return fig, axes

def calculate_scores(df: pd.DataFrame, target_columns: List[str]):
    scores = {}
    for target_column in target_columns:
        rmse = mean_squared_error(df[target_column], df[f"{target_column}_pred"]) ** (
            0.5
        )
        scores[f"{target_column}_rmse"] = rmse
    for target_column in target_columns:
        mae = mean_absolute_error(df[target_column], df[f"{target_column}_pred"])
        scores[f"{target_column}_mae"] = mae
    for target_column in target_columns:
        r2 = r2_score(df[target_column], df[f"{target_column}_pred"])
        scores[f"{target_column}_r2"] = r2
    return scores


def absolute_error_composition(df: pd.DataFrame):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    fig.subplots_adjust(wspace=0.2)
    big_df_errors = df.dropna().copy().reset_index()
    axis_fontsize = 16
    for i in [1,2]:
        abs_difference = (df.dropna()[f"ln_gamma_{i}"]-df.dropna()[f"ln_gamma_{i}_pred"]).abs()
        big_df_errors[f"abs_error_{i}"]= abs_difference.to_numpy()
        axes[i-1].scatter(df.dropna()["x(1)"], abs_difference, alpha=0.1, c = "#025b66")
        axes[i-1].set_xlabel("x", fontsize=axis_fontsize)
        axes[i-1].set_ylabel(f"Absolute Error $\ln\gamma_{i}$", fontsize=axis_fontsize)
        axes[i-1].set_title(f"$\ln\gamma_{i}$", fontsize=16)
        axes[i-1].tick_params(direction="in", which="both", labelsize=12)
    return fig, axes


def absolute_error_temperature(df: pd.DataFrame):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    fig.subplots_adjust(wspace=0.2)
    big_df_errors = df.dropna().copy().reset_index()
    axis_fontsize = 16
    for i in [1,2]:
        abs_difference = (df.dropna()[f"ln_gamma_{i}"]-df.dropna()[f"ln_gamma_{i}_pred"]).abs()
        big_df_errors[f"abs_error_{i}"]= abs_difference.to_numpy()
        axes[i-1].scatter(df.dropna()["temperature (K)"], abs_difference, alpha=0.1, c = "#025b66")
        axes[i-1].set_xlabel("Temperature (K)", fontsize=axis_fontsize)
        axes[i-1].set_ylabel(f"Absolute Error $\ln\gamma_{i}$", fontsize=axis_fontsize)
        axes[i-1].set_title(f"$\ln\gamma_{i}$", fontsize=16)
        axes[i-1].tick_params(direction="in", which="both", labelsize=12)
    return fig, axes

def absolute_error_temperature_histogram(df: pd.DataFrame):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    fig.subplots_adjust(wspace=0.2)
    big_df_errors = df.dropna().copy().reset_index()
    axis_fontsize = 16
    for i in [1,2]:
        abs_difference = (df.dropna()[f"ln_gamma_{i}"]-df.dropna()[f"ln_gamma_{i}_pred"]).abs()
        big_df_errors[f"abs_error_{i}"]= abs_difference.to_numpy()
        sns.histplot(data=big_df_errors, x="temperature (K)", y=f"abs_error_{i}", ax=axes[i-1])
        # axes[i-1].scatter(df.dropna()["temperature (K)"], abs_difference, alpha=0.1, c = "#025b66")
        # axes[i-1].set_xlabel("Temperature (K)", fontsize=axis_fontsize)
        # axes[i-1].set_ylabel(f"Absolute Error $\ln\gamma_{i}$", fontsize=axis_fontsize)
        # axes[i-1].set_title(f"$\ln\gamma_{i}$", fontsize=16)
        # axes[i-1].tick_params(direction="in", which="both", labelsize=12)
    return fig, axes


def calculate_activity_coefficients_polynomial(
    preds: pd.DataFrame, order=4
):
    # c1_0 is parameter 1 (zero-indexed) for molecule 1
    for i in range(2):
        preds[f"ln_gamma_{i+1}_pred"] = 0
        for j in range(order+1):
            x = preds["x(1)"] if i == 0 else 1.0-preds["x(1)"]
            preds[f"ln_gamma_{i+1}_pred"] += preds[f"c{j}_{i}_pred"]*x**j
    return preds

    
class VLEPredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""
    data_dir: Optional[str] = "data/"
    model_path: Optional[str] = None
    skip_prediction: bool = False
    skip_figures: bool = False
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
    results_path: str = "results"
    format_gammas: bool = True
    dataset: Literal["cosmo",  "cosmo-polynomial", "aspen"] = "cosmo"

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        self.data_input_dir = Path(self.data_dir) / "05_model_input"
        
        if "cosmo" in self.dataset:
            self.data_input_dir = self.data_input_dir / "cosmo"
        elif "aspen" == self.dataset:
            self.data_input_dir = self.data_input_dir / "aspen"
        self.results_path = Path(self.results_path)
        if not self.results_path.exists():
            os.makedirs(self.results_path)
        self.output_path = self.results_path / "07_model_output" / self.dataset
        if not self.output_path.exists():
            os.makedirs(self.output_path)
        self.reporting_dir = self.results_path / "08_reporting" / self.dataset
        if not self.reporting_dir.exists():
            os.makedirs(self.reporting_dir)
        super().process_args()



def evaluate():
    logger = setup_logger()
    # Arguments
    args = VLEPredictArgs().parse_args()

    # Download models
    model_paths = {}
    model_run_ids = {
        "DG": "20mq61e5",
        "DG-TLCB": "3bll2ycq",
        # "DGP": "2ib287pj",
        # "DGP-TLCB": "33u9qckt",
        # "aspen_base": "3g7mpeqy",
        # "aspen_base_pretrained": "3msj6d4l",
        # "cosmo_pretrained_depth_4": "3dxpryr1"
    }
    if not args.skip_prediction:
        if args.model_path is not None:
            model_paths["model"] = args.model_path 
        else:
            for name, run_id in model_run_ids.items():
                if "polynomial" in name:
                    path = "fold_0/model_0/model.pt" 
                else:
                    path = "fold_0/model_0/model.pt"
                wandb_base_path = f"ceb-sre/vle/{run_id}"
                checkpoint_path = wandb.restore(path, run_path=wandb_base_path, root=args.output_path / name)
                model_paths[name] = str(checkpoint_path.name)
    else:
        model_paths = model_run_ids

    # Loop through different validation and test sets.
    sets = ["valid_cont", "valid_mix", "valid_indp", "test_indp", "test_mix"]
    all_scores = []
    for model_name, model_path in model_paths.items():
        args.checkpoint_paths = [model_path]
        for predict_set in sets:
            # Paths
            if "polynomial" in model_name:
                args.test_path = args.data_input_dir /  f"{predict_set}_polynomial.csv"
                args.features_path = [args.data_input_dir / f"{predict_set}_polynomial_temperature.csv"]
                
            else:
                args.test_path = args.data_input_dir /  f"{predict_set}.csv"
                args.features_path = [args.data_input_dir / f"{predict_set}_features.csv"]
            args.preds_path = args.output_path / f"{model_name}_{predict_set}_preds.csv"
 
            # Make predictions
            if not args.skip_prediction:
                # Make predictions
                preds = make_predictions(args)

            # Read back in test predictions data
            preds = pd.read_csv(args.preds_path)
            if args.skip_prediction:
                args.target_columns = [col for col in preds.columns if col not in args.smiles_columns]
            preds = preds.rename(columns=lambda t: f"{t}_pred")
            truth = pd.read_csv(args.test_path)
            if len(args.features_path) > 0:
                features = pd.read_csv(args.features_path[0])
                big_df = pd.concat([truth, features, preds], axis=1)
            else:
                big_df = pd.concat([truth, preds], axis=1)

            # Calculate polynomial activity coefficients if needed
            if "polynomial" in model_name:
                test_df = pd.read_csv(args.data_input_dir /  f"{predict_set}.csv")
                features_df = pd.read_csv(args.data_input_dir / f"{predict_set}_features.csv")
                df = pd.concat([test_df, features_df], axis=1)
                big_df = df.merge(big_df, on=["smiles_1", "smiles_2", "temperature (K)"], how='left')
                big_df = calculate_activity_coefficients_polynomial(big_df)


            if args.drop_na:
                big_df = big_df.dropna()

            logger.info(f"Size of {predict_set} for {model_name}: {big_df.shape[0]}")
            # Calculate scores
            scores = calculate_scores(big_df, ["ln_gamma_1", "ln_gamma_2"])
            scores.update({
                "model_name": model_name,
                "holdout_set": predict_set
            })
            all_scores.append(scores)
            with open(args.output_path / f"{model_name}_{predict_set}_scores.json", "w") as f:
                json.dump(scores, f)

            # Plots
            if not args.skip_figures:
                #Parity plot
                fig, _ = parity_plot(big_df, ["ln_gamma_1", "ln_gamma_2"], format_gammas=args.format_gammas)
                fig.savefig(args.reporting_dir / f"{model_name}_{predict_set}_parity_plot.png", dpi=300)

                # Absolute error vs composition
                fig, _ = absolute_error_composition(big_df)
                fig.savefig(args.reporting_dir / f"{model_name}_{predict_set}_absolute_error_vs_composition.png", dpi=300)
                fig, _ = absolute_error_temperature(big_df)
                fig.savefig(args.reporting_dir / f"{model_name}_{predict_set}_absolute_error_vs_temperature.png", dpi=300)
    
    # Write out scores in publication format
    scores_df = pd.DataFrame(all_scores).round(4)
    scores_df = scores_df.sort_values(by="holdout_set").set_index(["holdout_set", "model_name"])
    scores_df.to_csv(args.reporting_dir / "scores.csv",)
    latex_table = scores_df.to_latex()
    with open(args.reporting_dir / "latex_table.txt", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    evaluate()