import pandas as pd
import numpy as np
from lmfit import minimize, Parameters

import ray
from ray.exceptions import GetTimeoutError

from tqdm.auto import tqdm, trange
import typer

from datetime import datetime as dt
import logging
import json
from pathlib import Path


BATCH_SIZE = 1000

def setup_logger(log_filename: str ="nrtl_fitting.log"):
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


def nrtl_model(alpha_12: float, tau_12: float, tau_21: float, x1: float, x2: float, alpha_21: float=None):
    if alpha_21 is None:
        alpha_21 = alpha_12
    g_12 = np.exp(-alpha_12*tau_12)
    g_21 = np.exp(-alpha_21*tau_21)
    ln_gamma_1 = x2**2*(tau_21*(g_21/(x1+x2*g_21))**2+tau_12*g_12/(x2+x1*g_12)**2)
    ln_gamma_2 = x1**2*(tau_12*(g_12/(x2+x1*g_12))**2+tau_21*g_21/(x1+x2*g_21)**2)
    ln_gamma_1 = np.nan_to_num(ln_gamma_1, 0.0)
    ln_gamma_2 = np.nan_to_num(ln_gamma_2, 0.0)
    return ln_gamma_1, ln_gamma_2

def residual(params, data):
    # Parameters
    alpha_12 = params['alpha_12']
    alpha_21 = params['alpha_21']
    tau_12 = params['tau_12']
    tau_21 = params['tau_21']

    # Composition
    x = data["x(1)"]

    # Calculate activity coefficients
    ln_gammas_1, ln_gammas_2 = nrtl_model(
        alpha_12, tau_12, tau_21, x, 1.0 - x, alpha_21=alpha_21
    )

    # Calcualate residual
    residual_1 = ln_gammas_1 - data["ln_gamma_1"]
    residual_2 = ln_gammas_2 - data["ln_gamma_2"]
    norm = data[["ln_gamma_1", "ln_gamma_2"]].max(axis=0).max()
    residual = (residual_1 + residual_2)
    if residual.isna().any():
        print(params)
        raise ValueError()
    return residual.to_numpy()

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def fit_nrtl(data, method="leastsq", params=None, single_alpha_value=False):
    if params is None:
        params = Parameters()
        params.add('alpha_12', value=0.2, min=1e-4, max=1.0)
        params.add('alpha_21', value=0.2, min=1e-4, max=1.0)
        params.add('tau_12', value=100, min=-366, max=366)
        params.add('tau_21', value=100, min=-366, max=366)
        if single_alpha_value:
            params["alpha_21"].expr = "alpha_12"

    out = minimize(residual,  params, method=method, args=(data,))
    return out

@ray.remote
def fit_nrtl_ray(data, method="leastsq", params=None):
    start = dt.now()
    out = fit_nrtl(data, method, params)
    end = dt.now()
    return out, end-start

def evaluate_fit(data: pd.DataFrame, params: Parameters):
    x = data["x(1)"]
    alpha_12 = params['alpha_12'].value
    alpha_21 = params["alpha_21"].value
    tau_12 = params['tau_12'].value
    tau_21 = params['tau_21'].value
    ln_gammas_1, ln_gammas_2 = nrtl_model(alpha_12, tau_12, tau_21, x, 1.0 - x, alpha_21=alpha_21)
    ln_gamma_1_mae = mae(data["ln_gamma_1"], ln_gammas_1)
    ln_gamma_2_mae = mae(data["ln_gamma_2"], ln_gammas_2)
    return {
        "alpha_12": alpha_12,
        "alpha_21": alpha_21,
        "tau_12": tau_12,
        "tau_21": tau_21,
        "ln_gamma_1_mae": ln_gamma_1_mae,
        "ln_gamma_2_mae": ln_gamma_2_mae,
    }

def main(input_file: str, output_dir: str, batch_size: int= 1000, nrows: int=None):
    # Set up logging
    logger = setup_logger()

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown
    ray.init()

    # Read in data
    logger.info("Reading in data")
    df = pd.read_csv(input_file, nrows=nrows)

    # Run fitting using ray
    results_ray = {}
    n_tasks = 0
    logger.info("Submitting fitting jobs to ray cluster")
    start = dt.now()
    for name, data in tqdm(df.groupby(by=["smiles_1", "smiles_2", "temperature (K)"])):
        # Submit job for NRTL FITTING
        out_ref = fit_nrtl_ray.remote(data, method="shgo")

        # Update results
        result = {
            "smiles_1": name[0],
            "smiles_2": name[1],
            "temperature (K)": name[2],
        }
        results_ray.update({out_ref: result})
        n_tasks += 1

    # Get results in batches:
    n_batches = n_tasks // batch_size
    n_batches += 1 if n_tasks % batch_size != 0 else 0
    object_refs = list(results_ray.keys())
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Waiting and retrieving results for f{len(object_refs)}")
    for batch in trange(n_batches):
        # Select correct refes
        refs = object_refs[batch*batch_size:(batch+1)*batch_size]

        # Wait for results
        ready_refs, remaining_refs = ray.wait(
            refs, num_returns=len(refs)
        )

        # Get results
        results_ray_list = []
        for ready_ref in ready_refs:
            # Retrieve existing results
            result = results_ray[ready_ref]

            # Get fitting result
            out, elapsed = ray.get(ready_ref, timeout=4)
            result.update({"time": elapsed.total_seconds()})

            # Evaluation
            result.update(evaluate_fit(data, out.params))

            # Append to results
            results_ray_list.append(result)

        # Save results from batch
        ray_df = pd.DataFrame(results_ray_list)
        ray_df.to_csv(output_dir / f"batch_{batch}.csv")

    end = dt.now()
    elapsed = end-start
    logger.info(f"Fitting took {elapsed.total_seconds()/3600} hours in total.")

if __name__ == "__main__":
    typer.run(main)