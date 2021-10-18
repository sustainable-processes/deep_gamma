# Uplodad data to datastore
# Launch training run
from deep_gamma import DATA_PATH
from dagster import graph
from dagster_shell import create_shell_command_op


@graph
def upload_data_to_grid():
    model_input_path = DATA_PATH / "05_model_input"
    name = "cosmo-gammas"
    upload_data = create_shell_command_op(
        f"grid datastore create --source {model_input_path} --name {name}",
        name="upload_data_to_grid",
    )
    upload_data()


@graph
def run_training():
    python_command = """
    train.py $1 $2/combisolv.txt --dataset_type regression --split_type random --split_sizes 0.8 0.1 0.1 --save_smiles_splits --save_preds --smiles_columns "mol solvent" "mol solute" --target_columns "target Gsolv kcal" --number_of_molecules 2 --depth 4 --batch_size 50 --hidden_size 200 --ffn_num_layers 4 --ffn_hidden_size 500 --activation LeakyReLU --epochs 200 --max_lr 2e-4 --metric mse --extra_metrics r2 mae "${ARGS[@]:2}"
    
    """

    create_shell_command_op("grid run ")


if __name__ == "__main__":
    upload_data_to_grid.execute_in_process()
