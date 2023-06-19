from .cosmo_calculate import get_cosmo_search_df, CosmoCalculate
from tqdm import tqdm


if __name__ == "__main__":
    import argparse

    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_path",
        type=str,
        help="Path of the COSMObase db. By default, searches in standard location for platform.",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Size of batches. Defaults to 100.", default=100
    )
    parser.add_argument("--start_row", type=int, help="Row to start calculations on")
    parser.add_argument(
        "--n_rows",
        type=int,
        help="Number of rows to run calculation for. Defaults to total rows in COSMObase",
    )
    parser.add_argument(
        "--n_cores",
        help="Number of cores to run calculation for. Defaults to max available",
        default="max",
    )

    args = parser.parse_args()

    # Get df
    df = get_cosmo_search_df(path=args.db_path)

    rows_read = 0 if not args.start_row else args.start_row
    batch_size = args.batch_size
    batch_size = 100 if batch_size is None else batch_size
    n_rows = args.n_rows
    n_rows = df.shape[0] if n_rows is None else n_rows
    n_rows -= rows_read
    n_batches = n_rows // batch_size
    n_batches += 1 if batch_size % n_rows > 0 else 0

    calc_func = CosmoCalculate(
        "Boiling Point",
        property_name="Tboil",
        lookup_name="uniqueCode12",
        lookup_type=CosmoCalculate.UNICODE,
        background=True,
        n_cores=args.n_cores,
    )

    column_name = "boiling_point[C]"
    df[column_name] = 0
    calc_func.ct.searchName("ethanol")
    bar = tqdm(range(n_batches))
    for batch in bar:
        # Create computation calculations
        mols = [
            calc_func(row)
            for _, row in df.iloc[rows_read : rows_read + batch_size].iterrows()
        ]

        # Actually run calculations
        bar.set_description(f"Calculating using {calc_func.ct.getNCores()} cores")
        calc_func.ct.finishQueue(verbose=False)

        # Read results
        # for i, mol in enumerate(mols):
        #     df.at[rows_read+i,column_name] = mol.getProperty("Tboil")

        # #Write to disk every batch
        # df.to_csv("boiling_points.csv")

        # Increase read rows
        rows_read += batch_size
    bar.close()
