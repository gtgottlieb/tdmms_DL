"""
Module to evaluate the AI

How to run from a terminal:
    1. activate your environment
    2. run: py bep_evaluation.py <model or dataset> --material <NbSe2 or MoS2> --weights MoS2

"""

import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Set the log level of tensorflow, see bep_utils for more information.

import tensorflow as tf

from tdmcoco import (
    CocoConfig, 
    CocoDataset,
    evaluate_coco
)
from bep_utils import (
    load_train_val_datasets,
    load_train_val_datasets_tdmms,
    check_dir_setup,
    create_dir_setup,
)
from bep_data import bepDataset

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

category_mapping = {
    1: "Mono_",
    2: "Few_",
    3: "Thick_"
}

class EvaluationConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'evaluation')

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
    print(f"Folder '{DEFAULT_LOGS_DIR}' created.")

def evaluate_dataset(material: str) -> None:
    """
    Function to evaluate the dataset. Returns the amount of images and class counts.
    
    Args:
        - material: NbSe2 (from BEP) or Graphene, Mos2, BN, WTe2 (from TDMMS)

    Data directory should be setup as the following:
    ROOT_DIR/
        DL_2DMaterials/
            Dataset_DL_2DMaterials/
                <material>/
                    train/
                    val/
        data/
            annotations/ (.ndjson or .json)
                train.ndjson
                val.ndjson
            images/
                train/
                val/
    """
    if material == 'NbSe2':
        dataset_train, dataset_val = load_train_val_datasets(ROOT_DIR)
    else:
        dataset_train, dataset_val = load_train_val_datasets_tdmms(ROOT_DIR, material)

    all_cls_train = [[category_mapping[j['category_id']]+material for j in i['annotations']] for i in dataset_train.image_info]
    all_cls_train = [i for j in all_cls_train for i in j]
    all_cls_val = [[category_mapping[j['category_id']]+material for j in i['annotations']] for i in dataset_val.image_info]
    all_cls_val = [i for j in all_cls_val for i in j]

    print('')
    print('Image count')
    print('    Train: {} images, {} part'.format(len(dataset_train.image_ids), round(len(dataset_train.image_ids)/(len(dataset_train.image_ids)+len(dataset_val.image_ids)),2)))
    print('    Val: {} images, {} part'.format(len(dataset_val.image_ids), round(len(dataset_val.image_ids)/(len(dataset_train.image_ids)+len(dataset_val.image_ids)),2)))
    print('')
    print('Class count')
    print('    Train')
    for i in dataset_train.class_names[1:]:
        print('        {}: {} images, {} part'.format(i, all_cls_train.count(i), round(all_cls_train.count(i)/len(all_cls_train),2)))
    print('    Val')
    for i in dataset_val.class_names[1:]:
        print('        {}: {} images, {} part'.format(i, all_cls_val.count(i), round(all_cls_val.count(i)/len(all_cls_val),2)))
    
    return None

def evaluate_model(material: str, weights: str, weights_path: str) -> None:
    """
    Function to evaluate a model on a dataset. The evaluation consists of calculating the
    precision and recall for multiple IoU thresholds.

    Args:
        - material: NbSe2 (from BEP) or Graphene, Mos2, BN, WTe2 (from TDMMS)
        - weights: Graphene, Mos2, BN, WTe2 (from TDMMS)
        - weights_path: specific filename af a weights file, used if weights
                        argument is set to NbSe2

    Data directory should be setup as the following:
    ROOT_DIR/
        DL_2DMaterials/
            Dataset_DL_2DMaterials/
                <material>/
                    train/
                    val/
        data/
            annotations/ (.ndjson or .json)
                train.ndjson
                val.ndjson
            images/
                train/
                val/
        weights/
            <material>_mask_rcnn_tdm_120.h5
    """
    config = EvaluationConfig()
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    print('Running evaluation using the {} data and {} weights..'.format(material, weights))

    if weights != 'NbSe2':
        MODEL_PATH = os.path.join(ROOT_DIR, 'weights', weights.lower()+'_mask_rcnn_tdm_0120.h5')
    else:
        MODEL_PATH = os.path.join(ROOT_DIR, 'weights', weights_path)

    model.load_weights(MODEL_PATH, by_name=True)

    if material == 'NbSe2':
        dataset_val = bepDataset()
        coco = dataset_val.load_dir(os.path.join(ROOT_DIR, 'data'), 'val', reload_annotations=True, return_coco=True)
        dataset_val.prepare()
    else:
        dataset_val = CocoDataset()
        val_type = "val" 
        coco = dataset_val.load_coco(
            os.path.join(ROOT_DIR, 'DL_2DMaterials', 'Dataset_DL_2DMaterials', material),
            val_type,
            return_coco=True
        )
        dataset_val.prepare()

    print("Running evaluation on {} images.".format(len(dataset_val.image_ids)))
    evaluate_coco(model, dataset_val, coco, "bbox")

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate model and dataset'
    )
    parser.add_argument(
        'command',
        help='model or dataset'
    )
    parser.add_argument(
        '--material', 
        required=False,
        default='NbSe2',
        help='NbSe2 or MoS2'
    )
    parser.add_argument(
        '--weights', 
        required=False,
        default='MoS2',
        help='NbSe2 or MoS2'
    )
    parser.add_argument(
        '--weights_path', 
        required=False,
        default='nbse2_from_mos2_images_20_epochs_111.h5',
        help='File name of the weights file'
    )

    args = parser.parse_args()

    check_dir_setup(ROOT_DIR, 0.7)

    if args.command == 'dataset':
        evaluate_dataset(args.material)
    
    if args.command == 'model':
        evaluate_model(args.material, args.weights, args.weights_path)