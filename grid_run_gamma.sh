# Local:  grid_run_gamma.sh --data_dir data/05_model_input

# Grid:  grid run --instance_type g4dn.xlarge grid_run_gamma.sh --data_dir grid:cosmo-gammas:1 --experiment_name cosmo_gammas

#### Login into wandb
wandb login eddd91debd4aeb24f212695d6c663f504fdb7e3c

ARGS=("$@")
python deep_gamma/ops/chem.py \
    $1 $2 \
    --mpn_shared --dataset_type regression \
    --split_type custom --cache_cutoff 1e9 \
    --number_of_molecules 2 --smiles_columns "smiles_1" "smiles_2" \
    --target_columns "ln_gamma_1" "ln_gamma_2" \
    --epochs 100 \
    --save_preds \
    --metric mse --extra_metrics r2 mae \
    "${ARGS[@]:2}"
