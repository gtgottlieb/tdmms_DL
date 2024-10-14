"""Module to evaluate the AI"""

import os
import sys

from bep_data import bepDataset
from tdmcoco import evaluate_coco

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from mrcnn import model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate_model():
    # Validation dataset
    dataset_val = bepDataset()
    val_type = "val" 
    coco = dataset_val.load_dir(os.path.join(ROOT_DIR, 'data'), 'val', return_coco=True)
    dataset_val.prepare()
    print("Running COCO evaluation on {} images.".format(len(dataset_val.image_ids)))
    model.keras_model.save("mrcnn_eval.h5")
    json_string = model.keras_model.to_json()
    with open('mrcnn_eval.json', 'w') as f:
        f.write(json_string)
        
    evaluate_coco(model, dataset_val, coco, "bbox")


if __name__ == '__main__':
    evaluate_model()