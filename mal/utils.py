"""Module with used function for Labelbox.com"""
import os
import numpy as np
import skimage
import json

from skimage.measure import find_contours
from matplotlib import patches
from mrcnn import utils

def load_image(path: str):
    """Load a specified image and return a [H,W,3] Numpy array."""
    image = skimage.io.imread(path)

    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)

    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    return image

def resize_image(image, config):
    """Resize an image according to configurations."""
    image, _, _, _, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE
    )

    return image
    
def extract_annotations(
    boxes,
    masks,
    class_ids, 
    class_names,
    scores=None,
) -> list:
    """
    Function to extract the polygon annotations from Mask RCNN detections.

    Args:
        - boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        - masks: [height, width, num_instances]
        - class_ids: [num_instances]
        - class_names: list of class names of the dataset
        - scores: (optional) confidence scores for each box
    
    Returns:
        - annotations: list with polygons
    """

    annotations = []

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to extract *** \n")
        return None
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]


    for i in range(N):
        instance = {
            'name': '',
            'polygon': []
        }

        if not np.any(boxes[i]):
            continue

        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label

        instance['name'] = label

        mask = masks[:, :, i]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = patches.Polygon(verts, facecolor="none")

            p = p.get_xy()
            p = [{'x': i[0], 'y': i[1]} for i in p]
            instance['polygon'] += p

        annotations.append(instance)
    
    return annotations

def create_annotations_folder(data: str, ROOT_DIR: str, overwrite: bool=False) -> None:
    annotations_folder = os.path.join(ROOT_DIR, 'mal', data)

    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
        print(f"Folder '{annotations_folder}' created.")
    elif not overwrite:
        print(f"Folder: '{annotations_folder}' already exists, set 'overwrite' to True if you want to overwrite the stored annotations.")
        raise AssertionError
    else:
        print(f"Folder: '{annotations_folder}' already exists, overwriting..")

    return None

def store_annotations(external_id: str, annotations: list, data: str, ROOT_DIR: str) -> None:
    annotations_folder = os.path.join(ROOT_DIR, 'mal', data)
    file_name = external_id.split('.')[0] + '.json'

    with open(os.path.join(annotations_folder, file_name), 'w+') as f:
        json.dump(annotations, f)

    # with open('some_file.json') as fin:
    #     some_strings = json.load(fin)

    return None