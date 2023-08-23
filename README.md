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


Eval
```bash
python deep_gamma/ops/eval.py --results_path=data/ --drop_na
```


