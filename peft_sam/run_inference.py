import os
from pathlib import Path

import yaml
import shutil
from glob import glob

from cellseg1.data.utils import read_mask_to_numpy, resize_mask
from cellseg1.metrics import average_precision
from cellseg1.predict import predict_config

from micro_sam.evaluation.evaluation import run_evaluation

ALL_DATASETS = {'covid_if': 'lm', 'orgasegment': 'lm', 'gonuclear': 'lm', 'mitolab_glycolytic_muscle': 'em_organelles',
                'platy_cilia': 'em_organelles', 'hpa': 'lm', 'livecell': 'lm'}


def adjust_config(config, model, dataset, finetuned=True):
    if finetuned:
        config["model_path"] = f"/scratch/usr/nimcarot/sam/experiments/peft/checkpoints/{model}/cellseg1/{dataset}/best.pt"
    else:
        config["model_path"] = f"/user/teuber5/u12094/.cache/micro_sam/models/{model}"
    config["data_dir"] = f"/scratch/usr/nimcarot/data/{dataset}/slices"
    config["result_dir"] = f"/scratch/usr/nimcarot/sam/peft/cellseg1/{model}/{dataset}/"


def run_inference(config):
    pred_masks = predict_config(config, save=True)

    true_masks_path = Path(config["data_dir"]) / "test/masks"
    true_masks_files = sorted(list(Path(true_masks_path).iterdir()))
    true_masks = [read_mask_to_numpy(i) for i in true_masks_files]
    true_masks = [resize_mask(i, config["resize_size"]) for i in true_masks]
    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=0.5)
    score = ap.mean(axis=0)[0]
    print(f"mAP@0.5: {score}")

    pred_paths = sorted(glob(os.path.join(config["result_dir"], "pred_masks", "*")))
    gt_paths = sorted(glob(os.path.join(config["data_dir"], "test", "masks", "*"))) 
    save_path = os.path.join(config["result_dir"], "results", "amg_opt.csv")

    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


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
    new_config_path = os.path.join(config_dir, f"{dataset}_{model}.yaml")

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

    for dataset, roi in ALL_DATASETS.items():
        models = ["vit_b", f"vit_b_{roi}"] if dataset != "livecell" else ["vit_b"]
        for model in models:

            config = copy_and_modify_config(config_file, model, dataset, finetuned=True) 
            print(config)
            run_inference(config)

            config = copy_and_modify_config(config_file, model, dataset, finetuned=False) 
            print(config)
            run_inference(config)