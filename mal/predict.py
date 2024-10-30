"""
Module to make predictions and store them in a .ndjson file.

How to run from a terminal:
    1. activate your AI environment
    2. run $ py mal/predict.py <dataset>
        Optional arguments:
            --weights <weights filename>
        Example:
            $ py mal/upload.py batch4 --weights nbse2_from_mos2_images_20_epochs_111.h5
"""

import os
import sys
import argparse
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from utils import (
    load_image,
    extract_annotations,
    create_annotations_folder,
    store_annotations,
    sieve_annotations
)
from dataset import malDataset
from tdmms.tdmcoco import CocoConfig

from mrcnn import model as modellib

class malConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'mal')

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
    print(f"Folder '{DEFAULT_LOGS_DIR}' created.")

def predict(data: str, weights: str=None) -> None:
    """
    Function to make predictions and store them.
    First predictions are made and then the polygon
    annotations are extracted.
    The annotations are stored per image in a .json file.
    The .json files are stored at:
        ROOT_DIR/mal/<data>/
    
    Args:
        - data: which folder in the data/ folder to use.
        - weights: which weights to use for predictions, default is
            'nbse2_from_mos2_images_20_epochs_111.h5'.
    """
    config = malConfig()
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    if not weights:
        weights = 'nbse2_from_mos2_images_20_epochs_111.h5'

    print(f'Loading {weights} weights file..')

    MODEL_PATH = os.path.join(ROOT_DIR, 'weights', weights)
    model.load_weights(MODEL_PATH, by_name=True)

    dataset = malDataset(malConfig)
    dataset.load_dir(os.path.join(ROOT_DIR, 'data'), data)

    create_annotations_folder(data, ROOT_DIR, overwrite=True)

    print('Making and storing predictions per image..')

    for image_info in tqdm(dataset.image_info):
        external_id = image_info['path'].split('\\')[-1]
        image = load_image(image_info['path'])

        results = model.detect([image])
        results = results[0]

        annotations = extract_annotations(
            results['rois'],
            results['masks'],
            results['class_ids'],
            ['','Mono', 'Few','Thick'],
            results['scores'],
            sieve_amount=10
        )

        store_annotations(external_id, annotations, data, ROOT_DIR)
    
    return None
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make and store model predictions'
    )

    parser.add_argument(
        'command', 
        help='Which dataset to use.'
    )

    parser.add_argument(
        '--weights', 
        required=False,
        default=None,
        help='Weights filename'
    )

    args = parser.parse_args()

    predict(args.command, args.weights)