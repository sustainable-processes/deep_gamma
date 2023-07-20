"""Fit polynomials """
import pandas as pd
import numpy as np
from lmfit import Parameters
from lmfit.model import ModelResult
from lmfit.models import PolynomialModel

import ray

from tqdm.auto import tqdm, trange
import typer
from typing import Optional, List

from datetime import datetime as dt
import logging
from pathlib import Path


BATCH_SIZE = 1000
POLYNOMIAL_DEGREE = 4


app = typer.Typer()

def setup_logger(log_filename: str = "polynomial_fitting.log"):
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


def fit_polynomials(data: pd.DataFrame, degree: Optional[int] = 4) -> List[ModelResult]:
    logger  = logging.getLogger(__name__)
    results = []
    for i in [1, 2]:
        model = PolynomialModel(degree=degree, nan_policy="omit")
        x = data["x(1)"].astype(float).to_numpy()
        y = data[f"ln_gamma_{i}"].astype(float).to_numpy()
        try:
           params = model.guess(y, x=x)
           result = model.fit(y, params, x=x)
           results.append(result)
        except TypeError:
           logger.warning("Fitting failed")
    return results


def flatten_list(starting_list: list):
    final_list = []
    for l in starting_list:
        if type(l) not in [list, tuple]:
            final_list.append(l)
        else:
            final_list.extend(flatten_list(l))
    return final_list


@ray.remote
def fit_polynomials_ray(data, **kwargs):
    start = dt.now()
    results = fit_polynomials(data, **kwargs)
    end = dt.now()
    return results, end - start


def get_params(model_results: List[ModelResult]):
    params_dict = {}
    for i, model_result in enumerate(model_results):
        params = model_result.best_values
        params = {f"{param}_{i}": v for param, v in params.items()}
        params_dict.update(params)
        params_dict.update({
            f"aic_{i}": model_result.aic,
            f"bic_{i}": model_result.bic,
            f"chisqr_{i}": model_result.result.chisqr
        })
    return params_dict

@app.command("fit")
def main(input_file: str, output_dir: str, batch_size: int = 1000, nrows: int = None):
    # Set up logging
    logger = setup_logger()

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown
    ray.init()

    # Read in data
    logger.info("Reading in data")
    df = pd.read_parquet(input_file)

    # Run fitting using ray
    results = []
    object_refs = []
    n_tasks = 0
    logger.info("Submitting fitting jobs to ray cluster")
    start = dt.now()
    for name, data in tqdm(df.groupby(by=["smiles_1", "smiles_2", "temperature (K)"])):
        if data.shape[0] < 4:
            logger.warning(f"""{data["names_1"].iloc[0]},{data["names_2"].iloc[0]} only have {len(data)} examples, skipping.""")
            continue
        # Submit job for NRTL FITTING
        out_ref = fit_polynomials_ray.remote(data, degree=POLYNOMIAL_DEGREE)

        # Update results
        results.append(
            {
                "smiles_1": name[0],
                "smiles_2": name[1],
                "temperature (K)": name[2],
            }
        )
        object_refs.append(out_ref)
        n_tasks += 1

    
    # Get results in batches:
    n_batches = n_tasks // batch_size
    n_batches += 1 if n_tasks % batch_size != 0 else 0
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Waiting and retrieving results for {len(object_refs)} datsets")
    for batch in trange(n_batches):
        # Select correct refes
        refs = object_refs[batch * batch_size : (batch + 1) * batch_size]
        some_results = results[batch * batch_size : (batch + 1) * batch_size]

        # Wait for results
        ray.wait(refs, num_returns=len(refs))

        # Get results
        results_ray_list = []
        for ref, result in zip(refs, some_results):
            # Get fitting result
            model_results, elapsed = ray.get(ref, timeout=4)
            result.update({"time": elapsed.total_seconds()})

            # Evaluation
            if len(model_results) > 0:
                result.update(get_params(model_results))

            # Append to results
            results_ray_list.append(result)
            
        # Save results from batch
        ray_df = pd.DataFrame(results_ray_list)
        ray_df.to_csv(output_dir / f"batch_{batch}.csv")

    end = dt.now()
    elapsed = end - start
    logger.info(f"Fitting took {elapsed.total_seconds()/3600} hours in total.")

@app.command()
def split(input_dir: str, split_dir: str,  output_dir: str):
    input_dir = Path(input_dir)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger()

    all_polynomial_batches = input_dir.glob("batch_*.csv")
    logger.info("Reading in polynomial fit data")
    all_polynomial_batches = list(all_polynomial_batches)
    dfs = [pd.read_csv(p) for p in tqdm(all_polynomial_batches)]
    polynomial_df = pd.concat(dfs)

    split_types = ["indp", "mix"]
    split_names = ["valid", "test"]
    holdout_splits = [f"{split_name}_{split_type}" for split_name in split_names for split_type in split_types]
    splits = ["train"] + holdout_splits
    logger.info("Creating splits")
    for split in tqdm(splits):
        split_df = pd.read_csv(split_dir / f"{split}.csv")
        features_split_df = pd.read_csv(split_dir / f"{split}_features.csv")
        split_df = pd.concat([split_df, features_split_df], axis=1)
        polynomial_split_df = polynomial_df.merge(
            split_df, 
            on=["smiles_1", "smiles_2", "temperature (K)"],
            how="inner"
        ).drop_duplicates()
        polynomial_split_df.to_csv(output_dir /f"{split}_polynomial.csv" )




if __name__ == "__main__":
    # typer.run(main)
    app()


# polynomial_df = pd.read_csv("data/03_primary/polynomial_good_fit.csv")
# df = pd.read_csv("data/data_no_features.csv")
# features = pd.read_csv("data/features.csv")
# df = pd.concat([df, features], axis=1)

# p = Path("data/05_model_input")
# indices = {file.stem.rpartition("_indices")[0]: np.loadtxt(file) for file in p.glob("*_indices.txt")}
# dfs = {
#     name: df.iloc[inds].drop(
#         ["ln_gamma_1", "ln_gamma_2", "x(1)"], axis=1
#     ) 
#     for name, inds in indices.items()
# }
# dfs_polynomials = {
#     name: polynomial_df.merge(
#         df_split, 
#         on=["smiles_1", "smiles_2", "temperature (K)"],
#         how="inner"
#     ).drop_duplicates()
#     for name, df_split in dfs.items()
# }

# main_columns = [
#     "smiles_1","smiles_2",
#     "c0_0","c1_0","c2_0","c3_0","c4_0",
#     "c0_1","c1_1","c2_1","c3_1","c4_1"
# ]
# for name, df_split in dfs_polynomials.items():
#     df_split[main_columns].to_csv(f"data/05_model_input/{name}_polynomial.csv", index=False)
#     df_split["temperature (K)"].to_csv(f"data/05_model_input/{name}_polynomial_temperature.csv", index=False)

    