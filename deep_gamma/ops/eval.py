from chemprop.train import make_predictions
from chemprop.args import PredictArgs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from pathlib import Path
from typing import List


def parity_plot(df, target_columns):
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


class VLEPredictArgs(PredictArgs):
    skip_prediction: bool = False
    target_columns: List[str]
    drop_na: bool = False


def evaluate():
    # Arguments
    args = VLEPredictArgs().parse_args()

    if not args.skip_prediction:
        # Make predictions
        preds = make_predictions(args)

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
    fig.savefig(Path(args.preds_path).parent / "parity_plot", dpi=300)


if __name__ == "__main__":
    evaluate()
