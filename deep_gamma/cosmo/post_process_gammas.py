"""
Post processing of results from calculate_gammas script
"""
import pandas as pd
from tqdm.auto import tqdm, trange
import re
import glob
import pathlib


def process_dir(input_dir, output_dir, batch_size: int = 1000):
    dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    tab_files = glob.glob(str(dir / "*.tab"))
    num_files = len(tab_files)
    num_batches = num_files // batch_size
    if num_batches == 0:
        num_batches = 1
    elif num_files % batch_size != 0:
        num_batches += 1
    for batch in trange(num_batches):
        dfs = [
            process_file(tab_file)
            for tab_file in tqdm(
                tab_files[batch * batch_size : (batch + 1) * batch_size], leave=False
            )
        ]
        df = pd.concat(dfs)
        df.to_csv(output_dir / f"batch_{batch}.csv")


def process_file(filepath: pathlib.Path):
    cas_numbers = filepath.split("__")
    cas_numbers[1] = cas_numbers[1].rstrip("_ct.tab")

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Compound names
    names = re.findall(r"(?<=\s\s\s\d\s)[\w\(\)\,\-]+(?=\s+)", "".join(lines[:8]))

    # Read through the file
    num_lines = len(lines)
    num_blocks = num_lines // 8
    records = [process_block(lines[i * 8 : (i + 1) * 8]) for i in range(num_blocks)]

    # Create data frame
    df = pd.DataFrame(records)
    df["cas_number_1"] = cas_numbers[0]
    df["cas_number_2"] = cas_numbers[1]
    if len(names) == 2:
        df["names_1"] = names[0]
        df["names_2"] = names[1]
    else:
        df["names_1"] = None
        df["names_2"] = None

    return df


def process_block(block):
    # Get temperature
    check = re.search(r"(?<=T= )\d+.\d+", block[2])
    if check:
        temperature = check[0]
    else:
        raise ValueError(f"No temperature found: {block[2]}")

    # Get composition 1
    check = re.search(r"(?<=x\(1\)= )\d+.\d+", block[2])
    if check:
        composition_1 = check[0]
    else:
        composition_1 = 0.0

    # Get composition 2
    check = re.search(r"(?<=x\(2\)= )\d+.\d+", block[2])
    if check:
        composition_2 = check[0]
    else:
        composition_2 = 0.0

    # Get activity coefficients
    ln_gamma_1 = re.search(r"\-?\d+.\d+$", block[6])[0]
    ln_gamma_2 = re.search(r"\-?\d+.\d+$", block[7])[0]

    return {
        "temperature (K)": temperature,
        "x(1)": composition_1,
        "x(2)": composition_2,
        "ln_gamma_1": ln_gamma_1,
        "ln_gamma_2": ln_gamma_2,
    }


if __name__ == "__main__":
    import argparse

    # Set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default=".", type=str, help="Directory with .tab files"
    )
    parser.add_argument(
        "--output_dir", default=".", type=str, help="Directory to put the dataframes"
    )
    parser.add_argument(
        "--batch_size", default=1000, type=int, help="Size of processing batches"
    )
    args = parser.parse_args()
    process_dir(**vars(args))
