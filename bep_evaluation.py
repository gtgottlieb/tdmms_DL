"""Module to evaluate the AI"""

import os
import sys
import argparse

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tdmcoco import (
    evaluate_coco,
    CocoConfig, 
    CocoDataset
)
from bep_utils import (
    load_train_val_datasets,
    check_dir_setup,
    create_dir_setup,
)
from bep_data import bepDataset

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

category_mapping = {
    1: "Mono_Graphene",
    2: "Few_Graphene",
    3: "Thick_Graphene"
}

class EvaluationConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'evaluation')
MODEL_PATH = os.path.join(ROOT_DIR, 'weights', 'mos2_mask_rcnn_tdm_0120.h5') # Graphene COCO+2D

def evaluate_dataset():
    dataset_train, dataset_val = load_train_val_datasets(ROOT_DIR)

    all_cls_train = [[category_mapping[j['category_id']] for j in i['annotations']] for i in dataset_train.image_info]
    all_cls_train = [i for j in all_cls_train for i in j]
    all_cls_val = [[category_mapping[j['category_id']] for j in i['annotations']] for i in dataset_val.image_info]
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

def evaluate_model(data, model: str='MoS2'):
    config = EvaluationConfig()
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    print('Running evaluation using the {} data and {} weights..'.format(data, model))

    if model == 'MoS2':
        model.load_weights(MODEL_PATH, by_name=True)

    if data == 'bep':
        dataset_val = bepDataset()
        coco = dataset_val.load_dir(os.path.join(ROOT_DIR, 'data'), 'val', reload_annotations=True, return_coco=True)
        dataset_val.prepare()

    if data == 'tdmms':
        dataset_val = CocoDataset()
        val_type = "val" 
        coco = dataset_val.load_coco(
            os.path.join(ROOT_DIR, 'DL_2DMaterials', 'Dataset_DL_2DMaterials', 'MoS2'),
            val_type,
            return_coco=True
        )
        dataset_val.prepare()

    print("Running evaluation on {} images.".format(len(dataset_val.image_ids)))
    evaluate_coco(model, dataset_val, coco, "bbox")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate model and dataset'
    )
    parser.add_argument(
        'command',
        help='model or dataset'
    )
    parser.add_argument(
        '--data', 
        required=False,
        default='bep',
        help='bep or tdmms'
    )
    parser.add_argument(
        '--model', 
        required=False,
        default='MoS2',
        help='bep or MoS2'
    )
    args = parser.parse_args()

    check_dir_setup(ROOT_DIR, 0.7)

    if args.command == 'dataset':
        evaluate_dataset()
    
    if args.command == 'model':
        evaluate_model(args.data, args.model)