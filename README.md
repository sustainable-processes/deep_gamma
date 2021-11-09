## Local python
COSMO
```bash 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_base --artifact_name cosmo_base
```

COSMO polynomial
```bash 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_polynomial --artifact_name cosmo_polynomial --polynomial
```

Combisolv
```bash 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_base --artifact_name cosmo_base --combisolv
```

## Grid

Run combisolv model
``` bash
grid run --instance_type g4dn.xlarge --dependency_file requirements.txt  deep_gamma/ops/chem.py --data_dir grid:combisolv:10 --experiment_name combisolv_mpn_shared --artifact_name cosmo_mpn_shared --batch_size 50 --combisolv
```

Create datastore for COSMO
```bash
cd data/
tar -czvf model_input.tar.gz 05_model_input/*
grid datastore create --source model_input.tar.gz --name cosmo-gammas
```

Run base COSMO model
```bash
grid run --instance_type p3.2xlarge --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:cosmo-gammas:10 --experiment_name cosmo_base
```

Run polynomial model
``` bash
grid run --instance_type p3.2xlarge --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:cosmo-gammas:10 --experiment_name cosmo_polynomial --polynomial
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

* Notes on instances
    - g4dn.xlarge worked well for combisolv data
    - p2.xlarge is the cheapest machine on grid that has enough RAM for COSMO (61GB)