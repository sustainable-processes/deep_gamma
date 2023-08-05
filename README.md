# Installation

1. Clone the repository
    ```bash
    git clone https://github.com/sustainable-processes/deep_gamma.git
    ```
2. Get the data

    TO-DO

3. Initialize chemprop submodule

    ```bash
    git submodule init  
    git submodule update
    ```

# Data generation

See README in`deep_gamma/data/cosmo`

# Preprocessing


```bash
python deep_gamma/graphs/data_preprocessing.py 
```

To process polynomial data
```bash
python deep_gamma/ops/polynomial.py fit data/02_intermediate/cosmo_data.pq data/03_primary/polynomial
python deep_gamma/ops/polynomial.py split data/03_primary/polynomial data/05_model_input/cosmo data/05_model_input/cosmo
```

# Machine Learning 

## Local python 
COSMO
``` bash 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_base
```

``` bash 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_base_pretrained --wandb_checkpoint_frzn_run=2532qdqg
```

COSMO polynomial
```bash 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_polynomial  --dataset cosmo-polynomial
```

Combisolv
```bash 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_base --artifact_name cosmo_base --combisolv
```

Fingerprints
``` bash
python deep_gamma/ops/fingerprint.py --wandb_checkpoint_run 32vlsf8l --results_path data/ --dataset aspen
```
Then rename to `data/07_model_output/aspen` to `aspen_base_fingerints`

``` bash
python deep_gamma/ops/fingerprint.py --wandb_checkpoint_run 3s8jnyvi --results_path data/ --dataset aspen
```
Then rename to `data/07_model_output/aspen` to `aspen_pretrained_fingerints`


## Grid

Run combisolv model
``` bash
grid run --instance_type g4dn.xlarge --dependency_file requirements.txt  deep_gamma/ops/chem.py --data_dir grid:combisolv:10 --experiment_name combisolv_mpn_shared --artifact_name cosmo_mpn_shared --batch_size 50 --combisolv
```

Create datastore for COSMO
```bash
cd data/
tar -czvf model_input.tar.gz 05_model_input/*
grid datastore create --source model_input.tar.gz --name gammas
```

Run base COSMO model
```bash
grid run --instance_type p3.2xlarge --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:cosmo-gammas:10 --experiment_name cosmo_base
```

Run polynomial model
``` bash
grid run --instance_type p3.2xlarge --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:cosmo-gammas:12 --experiment_name cosmo_polynomial --polynomial
```

Run COSMO model with molecule weights
```bash
grid run --instance_type p3.2xlarge --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:cosmo-gammas:10 --experiment_name cosmo_molecule_weights --use_molecule_weights
```

To get pretrained model, add this cli option
```bash
--wandb_checkpoint_frzn_run 2532qdqg
```

Running hyperparameter sweep
1. Start wandb sweeep `wandb sweep sweep.yaml`
2. Start agent on grid:
    ```bash
    grid run --instance_type p2.xlarge  --dependency_file requirements.txt --datastore_name cosmo-gammas --datastore_version 10  --use_spot run_wandb_agent.sh
    ```

Evaluation

```bash
grid run --instance_type g4dn.xlarge --dependency_file requirements.txt deep_gamma/ops/eval.py --drop_na --data_dir grid:cosmo-gammas:12
```

Create datastore for aspen
```bash
cd data/
tar -czvf model_input.tar.gz 05_model_input/*
grid datastore create --source model_input.tar.gz --name aspen-gammas
```

Run base aspen model
```bash
grid run --instance_type p3.2xlarge --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:aspen-gammas:1 --experiment_name aspen_base
```

Run pretrained aspen model
```bash
grid run --instance_type p3.2xlarge --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:gammas:1 --experiment_name aspen_pretrained --wandb_checkpiont_frzn_run zn669uuj
```

* Notes on instances
    - g4dn.xlarge worked well for combisolv data
    - p3.2xlarge is the cheapest machine on grid that has enough RAM for COSMO (61GB)

