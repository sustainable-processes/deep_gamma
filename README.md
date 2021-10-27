## Local python

```python 
python deep_gamma/ops/chem.py --data_dir data/ --experiment_name cosmo_base --artifact_name cosmo_base
```

## Grid

Create datastore
```bash
cd data/
tar -czvf model_input.tar.gz 05_model_input/*
grid datastore create --source model_input.tar.gz --name cosmo-gammas
```

```bash
grid run --instance_type p2.xlarge  --dependency_file requirements.txt deep_gamma/ops/chem.py --data_dir grid:cosmo-gammas:7 --experiment_name cosmo_base --artifact_name cosmo_base
 ```