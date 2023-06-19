from cosmo_calculate import CosmoCalculate
import pandas as pd
from scipy.special import comb
import numpy as np
from tqdm import tqdm, trange
import pkg_resources
import logging


def calculate_combinations(n):
    # Calculate all (N, 2) combinations in terms of indices
    indices = np.arange(0, n)
    pairs = np.zeros([int(comb(n, 2)), 2])
    offset = 0
    for i in indices:
        r = n - i - 1
        a = np.tile(i, r)
        b = indices[i + 1 :]
        pairs[offset : offset + r, 0] = a
        pairs[offset : offset + r, 1] = b
        offset += r
    return pairs


def setup_logger(log_filename="calculate_gammas.log"):
    # Logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)

    # Std out handler
    std_handler = logging.StreamHandler()
    std_handler.setLevel(level=logging.ERROR)

    # create a file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(std_handler)
    return logger


def setup_notifications(logger):
    installed = {pkg.key for pkg in pkg_resources.working_set}
    notify_installed = True if "notify-run" in installed else False
    if not notify_installed:
        logger.info("Notify run is not installed so notifications will not be sent.")
        return False
    else:
        from notify_run import Notify

        notify = Notify()
        if notify.endpoint is None:
            notify_installed = False
            logger.info("Notify run not registered, so it will not be used.")
            return False
        return notify


def main(args, logger):
    # Read in molecules dataset
    df = pd.read_csv(args.csv_file)
    n_molecules = df.shape[0]

    # Calculate indices of all combinations of binary mixtures
    combinations = calculate_combinations(n_molecules)

    rows_read = 0 if not args.start_mixture_index else args.start_mixture_index
    batch_size = args.batch_size
    batch_size = 100 if batch_size is None else batch_size
    n_rows = args.n_mixtures
    n_rows = combinations.shape[0] if n_rows is None else n_rows
    n_rows -= rows_read
    n_batches = n_rows // batch_size
    n_batches += 1 if batch_size % n_rows > 0 else 0

    # Setup connection to COSMO-RS
    calc_func = CosmoCalculate(
        "Activity Coefficient",
        lookup_name=args.lookup_name,
        lookup_type=args.lookup_type,
        background=True,
        n_cores=args.n_cores,
        cores_per_job=args.n_cores_per_job,
    )
    # Run a random lookup to establish connection database
    calc_func.ct.searchName("ethanol")

    calc_options = dict(
        xmin=args.xmin,
        xmax=args.xmax,
        xstep=args.xstep,
    )

    # Boiling points
    bp_column = args.boiling_point_column
    if args.boiling_point_units == "C":
        df[bp_column] = df[bp_column] + 273.15

    # Notifications
    notify_installed = False
    notify = setup_notifications(logger)
    if notify:
        notify_installed = True

    logger.info("Starting calculations")
    try:
        bar = trange(n_batches)
        for batch in bar:
            for i, combo in enumerate(
                combinations[rows_read : rows_read + batch_size, :]
            ):
                row_1 = df.iloc[int(combo[0])]
                row_2 = df.iloc[int(combo[1])]
                bp_1 = row_1[bp_column]
                bp_2 = row_2[bp_column]
                try:
                    calc_func(
                        row_1,
                        row_2,
                        Tmin=min(bp_1, bp_2),
                        Tmax=max(bp_1, bp_2),
                        Tstep=5,
                        **calc_options,
                    )
                except ValueError as e:
                    logger.error(e)

                if i > 2 * (args.n_cores // args.n_cores_per_job):
                    calc_func.ct.waitQueue()

            bar.set_description(f"Calculating using {calc_func.ct.getNCores()} cores")
            # Actually run calculations
            calc_func.ct.finishQueue(verbose=False)
            rows_read += batch_size

            # Send notifications
            if notify_installed and batch % args.notification_freq == 0:
                notify.send(f"Just finished batch {batch}.")

            logger.info(f"Batch {batch} completed.")

    except KeyboardInterrupt:
        print(f"{rows_read} iterations have been submitted to the job queue.")
        print("Stopping calculations...")
        calc_func.ct.killQueue(verbose=False)
        print("Calculations stopped.")


if __name__ == "__main__":
    import argparse

    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file", type=str, help="CSV file with boiling points and compound names"
    )
    parser.add_argument(
        "--lookup_name",
        type=str,
        help="Name of the column for molecule identifiers CSV",
        default="casNumber",
    )
    parser.add_argument(
        "--lookup_type",
        type=str,
        help="Identifier used for looking up molecules. Can be CAS, NAME or UNICODE",
        default="UNICODE",
    )
    parser.add_argument(
        "--boiling_point_column",
        type=str,
        help="Column with boiling points of mixtures.",
        default="boilingPoint[K]",
    )
    parser.add_argument(
        "--boiling_point_units",
        type=str,
        help="Boiling point units. C or K (default)",
        default="K",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        help="Minimum for compositon grid. Defaults to 0.0",
        default=0.0,
    )
    parser.add_argument(
        "--xmax",
        type=float,
        help="Maximum for compositon grid. Defaults to 1.0",
        default=1.0,
    )
    parser.add_argument(
        "--xstep",
        type=float,
        help="Step size for compositon grid. Defaults to 0.1",
        default=0.1,
    )
    parser.add_argument(
        "--batch_size", type=int, help="Size of batches. Defaults to 100.", default=100
    )
    parser.add_argument(
        "--n_mixtures",
        type=int,
        help="Number of mixtures to run calculation for. Defaults to all possible combinations.",
    )
    parser.add_argument(
        "--n_cores",
        help="Number of cores accessible for calculations. Defaults to 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_cores_per_job",
        help="Number of cores per job. Defaults to 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--start_mixture_index",
        type=int,
        help="Iteration to start at. Useful for restarting previously interrupted calculations",
    )
    parser.add_argument(
        "--notification_freq",
        type=int,
        help="Frequency of notifications in terms of batches (i.e,. every how many batches). Default to 100",
        default=100,
    )
    args = parser.parse_args()

    # Set up Logging
    logger = setup_logger()

    main(args, logger)
