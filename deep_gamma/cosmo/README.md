# Calculation of Activitiy Coefficients using COSMO-RS

The goal here is to generate a large dataset of activity coefficients that can be used to train a model.

## COSMO-RS Setup

We use [COSMO-RS](https://www.3ds.com/products-services/biovia/products/molecular-modeling-simulation/solvation-chemistry/) to generate our initial pretraining dataset. You will need COSMO-RS installed with at least the COSMObase2020 add-on.

For initial installation:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip pandas tqdm
```
You need to have [COSMOpy](https://cosmologic-services.de/cosmopy/index.html) installed, a python wrapper for COSMO-RS. You can ask BIOVIA for a copy or, if you are a member of the Sustainable Reactions group on Github, you can install it as follows:
```bash
pip install git+https://github.com/sustainable-processes/COSMOpy.git
```
You will need you [SSH keys](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/connecting-to-github-with-ssh) set up with Github. 

With these things setup, you can now run any of the data generation scripts. For example:
```bash
# Just calculate 10 boiling points as a demo
python calculate_boiling_points.py --n_rows=10 --batch_size=10
```

## Running the calculations
The activity coefficients are calculated at 0.1 composition intervals over 5 evenly space temperatures between the boiling points of each compound. Therefore, you need a dataset of compounds with their boiling points.

Initially, I started with trying to calculate the activity coefficients for every binary pair of molecules in the COSMO-RS 2020 database (see `archive/create_QM_data.ipynb`). I predicted the boiling points using COSMO-RS (see `calculate_boiling_points.py`). However, I quickly realized that running these COSMO calculations would require a couple years (there were ~3.8 million pairs). Instead, I decided to use the 460 solvents from [this paper](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c9sc01844a). The solvents are available in `solvent_descriptors.csv`, which already includes the boiling points.

I ran the calculations on our 24 core server and the following arguments (somehow it worked with n_cores=30 below):

```bash 
python calculate_gammas.py --lookup_name="cas_number" --lookup_type="CAS" \
--boiling_point_column="boiling_point" --boiling_point_units="C" \
--n_cores=30 --n_cores_per_job=3 --batch_size=1000  \
--notification_freq=1 solvent_descriptors.csv 
```
The calculations took ~41 days to run (8 January 2021 - 9 March 2021).

## Running post-processing
Creates the `cosmo_batches` directory with the initial input to the ML pipeline for predicting activity coefficients directly. 

```bash
python post_process_gammas.py --output_dir cosmo_batches
```

## Fitting NRTL Parameters

We also try training a model to predict NRTL parameters. In order to do that, we need, fit the NRTL parameter to the COSMO-RS data.
This is a bit messy, the code expects a file outputted from the ML pipeline preprocess step called cosmo_data.csv.

```bash
python nrtl_fitting.py cosmo_data.csv nrtl_parameters
```