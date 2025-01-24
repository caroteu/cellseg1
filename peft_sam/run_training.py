import os
import argparse

import yaml
import shutil

from cellseg1.cellseg1_train import main


def adjust_config(config, model, dataset):
    config["model_path"] = f"/user/teuber5/u12094/.cache/micro_sam/models/{model}"
    config["data_dir"] = f"/scratch/usr/nimcarot/data/{dataset}/slices"
    config["result_dir"] = f"/scratch/usr/nimcarot/sam/peft/cellseg1/{model}/{dataset}/"
    config["result_pth_path"] = f"/scratch/usr/nimcarot/sam/experiments/peft/checkpoints/{model}/cellseg1/{dataset}/best.pt"
    config["train_image_dir"] = f"/scratch/usr/nimcarot/data/cellseg1/{dataset}/train/images"
    config["train_mask_dir"] = f"/scratch/usr/nimcarot/data/cellseg1/{dataset}/train/masks"


def copy_and_modify_config(example_config_path, dataset, model):
    """
    Copies the example config file, modifies it, and saves the modified version 
    in a subdirectory called 'configs' under the name 'dataset_model.yaml'.

    Args:
        example_config_path (str): Path to the example config file.
        save_dir (str): Directory where the modified config will be saved.
        dataset (str): Dataset name to update in the config.
        model (str): Model name to update in the config.
    """
    # Ensure the save directory exists
    config_dir = os.path.join("configs")
    os.makedirs(config_dir, exist_ok=True)

    # Define the new config path
    new_config_path = os.path.join(config_dir, f"train_{dataset}_{model}.yaml")

    # Copy the example config to the new location
    shutil.copy(example_config_path, new_config_path)

    # Load the config file to modify it
    with open(new_config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Modify the config (update dataset and model)
    adjust_config(config, model, dataset)

    # Save the modified config
    with open(new_config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return config


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_file = ("base_config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--model_type", "-m", type=str, required=True)
    args = parser.parse_args()

    config = copy_and_modify_config(config_file, args.dataset, args.model_type)
    print(config)

    model = main(config, save_model=True)
