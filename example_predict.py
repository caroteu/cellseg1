import os
from pathlib import Path

import yaml

from cellseg1.data.utils import read_mask_to_numpy, resize_mask
from cellseg1.metrics import average_precision
from cellseg1.predict import predict_config

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_file = ("./example_config.yaml")
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # suppose the data is at /data2/zhoupeilin/celldata/cellseg_blood/test/images
    config["data_dir"] = "/scratch/usr/nimcarot/data/covid_if/slices"
    # the predicted masks will be saved here
    config["result_dir"] = (
        "/scratch/usr/nimcarot/sam/peft/cellseg1/vit_b_lm/covid_if/"
    )

    print(config)
    pred_masks = predict_config(config, save=True, result_folder=config["result_dir"])

    true_masks_path = Path(config["data_dir"]) / "test/masks"
    true_masks_files = sorted(list(Path(true_masks_path).iterdir()))
    true_masks = [read_mask_to_numpy(i) for i in true_masks_files]
    true_masks = [resize_mask(i, config["resize_size"]) for i in true_masks]
    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=0.5)
    score = ap.mean(axis=0)[0]    

    print(f"mAP@0.5: {score}")
