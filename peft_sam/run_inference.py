import os
from pathlib import Path
import argparse

import yaml
import shutil
from glob import glob
import imageio.v3 as imageio

from cellseg1.data.utils import read_mask_to_numpy, resize_mask
from cellseg1.metrics import average_precision
from cellseg1.predict import predict_config

from micro_sam.evaluation.evaluation import run_evaluation

ALL_DATASETS = {'covid_if': 'lm', 'orgasegment': 'lm', 'gonuclear': 'lm', 'mitolab_glycolytic_muscle': 'em_organelles',
                'platy_cilia': 'em_organelles', 'hpa': 'lm', 'livecell': 'lm'}


def adjust_config(config, model, dataset):
    config["model_path"] = f"/user/teuber5/u12094/.cache/micro_sam/models/{model}"
    config["result_pth_path"] = f"/scratch/usr/nimcarot/sam/experiments/peft/checkpoints/{model}/cellseg1/{dataset}/best.pt"
    if dataset == "mitolab_glycolytic_muscle":
        config["data_dir"] = "/scratch/usr/nimcarot/data/mitolab/slices/glycolytic_muscle"
    elif dataset == "platy_cilia":
        config["data_dir"] = "/scratch/usr/nimcarot/data/platynereis/slices/cilia"
    else:
        config["data_dir"] = f"/scratch/usr/nimcarot/data/{dataset}/slices"

    result_dir = f"/scratch/usr/nimcarot/sam/experiments/peft/cellseg1/{model}/{dataset}"
    os.makedirs(result_dir, exist_ok=True)
    config["result_dir"] = result_dir


def run_inference(config):
    pred_masks = predict_config(config, save=True, result_folder=os.path.join(config["result_dir"], "pred_masks"))

    true_masks_path = Path(config["data_dir"]) / "test/masks"
    true_masks_files = sorted(list(Path(true_masks_path).iterdir()))
    true_masks = []
    for mask_path in true_masks_files:
        file_name = mask_path.stem
        mask = read_mask_to_numpy(mask_path)
        mask = resize_mask(mask, config["resize_size"])
        true_masks.append(mask)
        gt_dir = os.path.join(config["data_dir"], "for_cellseg1", "labels")
        # save resized masks for evaluation
        os.makedirs(gt_dir, exist_ok=True)
        imageio.imwrite(os.path.join(gt_dir, f"{file_name}.tif"), mask)

    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=0.5)
    score = ap.mean(axis=0)[0]
    print(f"mAP@0.5: {score}")

    pred_paths = sorted(glob(os.path.join(config["result_dir"], "pred_masks", "*")))
    gt_paths = sorted(glob(os.path.join(config["data_dir"], "for_cellseg1", "labels", "*"))) 
    save_path = os.path.join(config["result_dir"], "results", "amg_opt.csv")

    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def copy_and_modify_config(example_config_path, model, dataset):
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
    new_config_path = os.path.join(config_dir, f"inference_{dataset}_{model}.yaml")

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

    config = copy_and_modify_config(config_file, args.model_type, args.dataset)
    run_inference(config)
