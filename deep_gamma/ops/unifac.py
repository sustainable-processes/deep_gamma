import thermo.unifac as unifac
from chemicals import search_chemical
from typing import Dict, List
from thermo.unifac import UFIP, UFSG, UNIFAC
from deep_gamma import DATA_PATH
import pandas as pd
import numpy as np
from tqdm import tqdm
from deep_gamma.ops.eval import parity_plot, calculate_scores
import json

class BinaryUnifac:
    def __init__(self) -> None:
        unifac.load_group_assignments_DDBST()
        self.molecule_list = pd.read_csv("data/01_raw/aspen_molecule_list.csv")
    
    def get_unifac_groups(self, name: str):
        try:
            chemical = search_chemical(name)
        except ValueError:
            row = self.molecule_list[self.molecule_list["smiles"]==name]
            if len(row) > 0:
                chemical = search_chemical(row["cas_number"].iloc[0])
            else:
                raise ValueError('No molecule found')
        return unifac.DDBST_MODIFIED_UNIFAC_assignments.get(chemical.InChI_key)

    def gammas(self, names: List[str], x, temp):
        unifac_groups = [self.get_unifac_groups(name) for name in names]
        for unifac_group in unifac_groups:
            if unifac_group is None:
                return []
        xs = [[xp, 1.0-xp] for xp in x]
        return [
            UNIFAC.from_subgroups(
                chemgroups=unifac_groups, 
                T=temp, 
                xs=xp, 
                version=0, 
                interaction_data=UFIP,
                subgroups=UFSG
            ).gammas()
            for xp in xs
        ]


if __name__ == "__main__":
    bu = BinaryUnifac()
    data = pd.read_csv(DATA_PATH / "01_raw" / "aspen_data.csv")

    skip_predictions = True
    
    # Calculate activity coefficients using UNIFAC

    if not skip_predictions:
        gammas = []
        groups = data.groupby(["smiles_1", "smiles_2", "TRange"])
        bar = tqdm(groups, total=len(groups))
        for idx, group in bar:
            x = group["x1"]
            temp = idx[2]
            try:
                g =  bu.gammas([idx[0], idx[1]], x, temp)
                if len(g) < len(x):
                    continue
                t =  [
                    {
                        "smiles_1": idx[0],
                        "smiles_2": idx[1],
                        "TRange": temp,
                        "x1": x.iloc[i],
                        "ln_gamma_1_pred": np.log(g[i][0]),
                        "ln_gamma_2_pred": np.log(g[i][1]),
                    }
                    for i in range(len(x))
                ]
                gammas.extend(t)
            except ValueError or AttributeError:
                print(f"Failed for {idx[0]}, {idx[1]}")
            except KeyError: 
                print(f"Failed for {idx[0]}, {idx[1]}")
        df = pd.DataFrame(gammas)
        df.to_csv("data/07_model_output/aspen_unifac.csv")
    else:
        df = pd.read_csv("data/07_model_output/aspen_unifac.csv")
    

    df_new = data.merge(df, on=["smiles_1", "smiles_2", "TRange"], how="left").dropna()
    for i in [1,2]:
        for j in ["", "_pred"]:
            df_new[f"ln_gamma_{i}{j}"] = df_new[f"ln_gamma_{i}{j}"].astype(float)
    
    # Evaluation
    scores = calculate_scores(df_new, ["ln_gamma_1", "ln_gamma_2"])
    with open("data/07_model_output/aspen_unifac_scores.json", "w") as f:
        json.dump(scores, f)
    fig, ax = parity_plot(
        df_new, ["ln_gamma_1", "ln_gamma_2"], scores=scores, format_gammas=True
    )
    fig.savefig("data/08_reporting/aspen/unifac_parity_plot.png", dpi=300)
    