import thermo.unifac as unifac
from chemicals import search_chemical
from typing import Dict, List
from thermo.unifac import UFIP, UFSG, UNIFAC
from deep_gamma import DATA_PATH
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def dev_read_data_aspen():
    """Dev mode read data in. Problably should change this to use Dagster modes"""
    data = pd.read_csv(DATA_PATH / "01_raw" / "aspen_data.csv")
    molecule_list_df = pd.Series(
        pd.concat(
            [data["smiles_1"], data["smiles_2"]]
        ).unique()
    ).to_frame().rename(columns={0: "smiles"})
    return molecule_list_df, data



if __name__ == "__main__":
    bu = BinaryUnifac()
    _, data = dev_read_data_aspen()
    
    gammas = []
    groups = data.groupby(["smiles_1", "smiles_2", "TRange"])
    bar = tqdm(groups, total=len(groups))
    j = 0
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
                    "ln_gamma_1": g[i][0],
                    "ln_gamma_2": g[i][1],
                }
                for i in range(len(x))
            ]
            gammas.extend(t)
            j+=1
        except ValueError or AttributeError:
            print(f"Failed for {idx[0]}, {idx[1]}")
        except KeyError: 
            print(f"Failed for {idx[0]}, {idx[1]}")
    df = pd.DataFrame(gammas)
    df.to_csv("data/07_model_output/aspen_unifac.csv")