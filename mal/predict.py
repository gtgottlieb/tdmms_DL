"""Module to make predictions on data and store them in a .ndjson file"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from utils import (
    load_image,
    resize_image,
    extract_annotations,
    create_annotations_folder,
    store_annotations,
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

def predict(data: str, weights: str=None):
    config = malConfig()
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    if not weights:
        weights = 'nbse2_from_mos2_images_20_epochs_111.h5'

    MODEL_PATH = os.path.join(ROOT_DIR, 'weights', weights)
    model.load_weights(MODEL_PATH, by_name=True)

    dataset = malDataset(malConfig)
    dataset.load_dir(os.path.join(ROOT_DIR, 'data'), data)

    create_annotations_folder(data, ROOT_DIR, overwrite=True)

    for image_info in dataset.image_info:
        external_id = image_info['path'].split('\\')[-1]
        image = load_image(image_info['path'])
        # image = resize_image(image, config)

        results = model.detect([image])
        results = results[0]

        annotations = extract_annotations(
            results['rois'],
            results['masks'],
            results['class_ids'],
            ['','Mono', 'Few','Thick'],
            results['scores']
        )

        store_annotations(external_id, annotations, data, ROOT_DIR)
    
if __name__ == '__main__':
    predict('batch4')