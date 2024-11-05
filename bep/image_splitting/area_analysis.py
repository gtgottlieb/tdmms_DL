import os
import sys
import argparse
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROOT_DIR = os.path.abspath("../")
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

from mal.utils import (
    load_image,
    extract_annotations,
)
from tdmms.tdmcoco import CocoConfig

from mrcnn import model as modellib

class malConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'area')

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
    print(f"Folder '{DEFAULT_LOGS_DIR}' created.")


def area_analysis(weights: str = None):
    area_list = []
    count_list = []

    config = malConfig()
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    if not weights:
        weights = 'nbse2_from_mos2_images_20_epochs_111.h5'

    ROOT_DIR = os.path.abspath("../")

    MODEL_PATH = os.path.join(ROOT_DIR, 'weights', weights)
    model.load_weights(MODEL_PATH, by_name=True)

    try:
        for i in tqdm(os.listdir(os.path.join(ROOT_DIR, 'data', 'images', 'syn'))):
            # print(f'Image: {i}')

            area = i.split('_')[-1].split('.')[0]        
            image = load_image(os.path.join(ROOT_DIR, 'data', 'images', 'syn', i))

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
            
            if annotations:
                amount = len(annotations)
            else:
                amount = 0
            
            area_list.append(area)
            count_list.append(amount)
    except Exception:


if __name__ == '__main__':
    area_analysis()