"""Module that contains functions used for image splitting."""

import cv2
import os
import json
import shutil
import random
import numpy as np

from typing import Tuple, List
from shapely.geometry import Polygon, Point


def create_background(
    square_size: int,
    sample_image_path: str,
) -> Tuple[np.ndarray, int, int]:
    """
    Function to create a background for the split images.
    Make sure the upper left corner of the sample image is
    residue free and just the wafer.

    Args:
        - square_size: int = the size of the upper left square of the image that will be
            tiled to fill the entire image.
        - sample_image_path: str = path of the image of which the upper left corner
            is used to create a background.

    Returns:
        - np.ndarray
        - sample image width
        - sample image height

    TODO: Make a picture of a wafer without flakes or residue to use for this background.
    """

    sample_image = cv2.imread(sample_image_path)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    small_square = cv2.GaussianBlur(sample_image[:square_size, :square_size], (15,15), 0)

    tile_height, tile_width = small_square.shape[:2]
    num_tiles_y = sample_image.shape[0] // tile_height + 1
    num_tiles_x = sample_image.shape[1] // tile_width + 1
    tiled_background = np.tile(small_square, (num_tiles_y, num_tiles_x, 1))[:sample_image.shape[0], :sample_image.shape[1], :]
    tiled_background = cv2.GaussianBlur(tiled_background, (11, 11), 0)

    return tiled_background, sample_image.shape[1], sample_image.shape[0]

def get_random_bg(ROOT_DIR, zoom):
    bg_dir = os.path.join(ROOT_DIR, 'data', 'background', zoom)
    bg_images = os.listdir(bg_dir)

    image = random.choice(bg_images)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_images_to_split_from_dataset(image_info: List[dict], annotation_threshold: int) -> list:
    """
    Function to extract the images from a dataset that have more annotations than the annotation
    threshold. These will be split.

    Args:
        - image_info: str = the image information of a dataset, contains all annotations and filenames.
        - annotation_threshold: int = what the annotation threshold is.

    Returns:
        - images: list = list of all image names that have more annotations than the threshold.
    """
    images = []
    
    for i in image_info:
        annotation_count = len(i['annotations'])

        if annotation_count >= annotation_threshold:
            images.append(i['path'].split('\\')[-1])

    return images

def get_images_to_split_from_batches(ROOT_DIR:str, annotation_threshold: int = 15):
    """
    Function to extract the images from all batches that have more annotations than the annotation
    threshold. These will be split.

    Args:
        - annotation_threshold: int = what the annotation threshold is.

    Returns:
        - images: list = list of all image names that have more annotations than the threshold.

    Note: currently not used
    """
    batches = [i for i in os.listdir(os.path.join(ROOT_DIR, 'data', 'images')) if ('batch' in  i and i != 'batchsplit')]
    print('Found batches:',', '.join(batches))

    images = []
    for batch in batches:
        rows = []
        with open(os.path.join(ROOT_DIR, 'data', 'annotations', batch+'.ndjson')) as f:
            rows += [json.loads(l) for l in f.readlines()]
        
        for row in rows:
            annotations = len(list(row['data_row']['project'].values())[0]['labels']['annotations']['objects'])
            if annotations >= annotation_threshold:
                images.append(row['data_row']['external_id'])

    return images


def get_image_ids_positions(image_info: List[dict], images: List[str]) -> list:
    """
    Function to get the index of the extracted images in the image info list.

    Args:
        - image_info: List[dict] = list with information of all images
        - images: List[str] = images to get indexes from
    """
    all_image_ids = [i['id'] for i in image_info]
    image_ids = [i['id'] for i in image_info if any(j in i['path'] for j in images)]
    image_id_positions = [all_image_ids.index(i) for i in image_ids]

    return image_id_positions

def update_annotations_dict(
    ROOT_DIR: str,
    filename: str,
    annotations_dict: dict,
    idx: int,
    annotation: dict,
    image_id: int,
    width: int,
    height: int,
    bbox: List[float]
) -> dict:
    """
    Function to update the global batch split annotations dictionary.
    Used to add the loop annotation to the dictionary and to add the
    split image to the directory.

    Args:
        - filename: str = filename of the split image
        - annotations_dict: dict = the current annotations dict
        - idx: int = annotation id, used to create unique ids
        - annotation: dict = annotation to add to the annotations dict
        - image_id: int = split image id, used to create unique ids
        - width: int = image width
        - height: int = image height
        - bbox: List[float] = the bounding box of the annotations in the split image

    Returns:
        - annotations_dict: dict = updated annotations dict
    """

    flake_annotation = annotation.copy()
    flake_annotation['image_id'] = filename
    flake_annotation['id'] = int('{}{}'.format(flake_annotation['id'], idx))

    annotations_dict['annotations'].append(flake_annotation)
    annotations_dict['images'].append({
        "id": int('{}{}'.format(image_id, idx)),
        "path": os.path.join(ROOT_DIR, 'data', 'images', 'split', filename),
        "file_name": filename,
        "width": width,
        "height": height,
        "date_captured": "",
        "license": 1,
        "coco_url": "",
        "flickr_url": "",
        "flake_bbox": bbox,
    })

    return annotations_dict

def store_image(
    ROOT_DIR: str,
    filename: str,
    flake_image: np.ndarray,
) -> None:
    """Function to write / store a split image."""
    cv2.imwrite(
        os.path.join(ROOT_DIR, 'data', 'images', 'batchsplit', filename),
        cv2.cvtColor(flake_image, cv2.COLOR_RGB2BGR)
    )

    return None

def cut_out_flake(
    tiled_background: np.ndarray,
    bbox: tuple,
    image: np.ndarray,
    border: int,
):
    """
    Function to create the split image. Copies the background and adds the 
    final extended bbox.

    Args:
        - tiled_background: np.ndarray = the created background
        - bbox: tuple = the bbox that contains all annotations to cutout
        - image: np.ndarray = the original image
        - border: int = how much to extend the cut out bbox

    Returns:
        - split image with annotations: np.ndarray
    """
    x, y, w, h = [int(i) for i in bbox]
    borders = np.array([y-border, y+h+border, x-border, x+w+border]).clip(0)

    flake_image = tiled_background.copy()
    flake_image[borders[0]:borders[1], borders[2]:borders[3]] = image[borders[0]:borders[1], borders[2]:borders[3]]

    return flake_image

def bbox_to_coords(bbox):
    """Function to convert a bbox to coordinates."""
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)

def coords_to_bbox(coords):
    """Function to convert coordinates to a bbox"""
    x1, y1, x2, y2 = coords
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return (x, y, w, h)

# def check_overlap(annotation: dict, bbox: Polygon):
#     segmentation = annotation['segmentation']
#     for segment in segmentation:
#         polygon = Polygon(np.array(segment).reshape(-1, 2))
#         if bbox.intersects(polygon):
#             return True
#     return False

def check_overlap(annotation: dict, bbox: Polygon):
    """Function that checks the overlap between an annotation and a bbox."""
    segmentation = annotation['segmentation']
    for segment in segmentation:
        polygon = Polygon(np.array(segment).reshape(-1, 2))
        for point in polygon.exterior.coords:
            if bbox.contains(Point(point)):
                return True
    return False

def check_overlap_image(
    already_loaded_ids_image: List[int],
    already_loaded_ids_flake: List[int],
    image_info: dict,
    bbox: Polygon
):
    """
    Function that checks the overlap of a bbox with all other annotations in 
    the image. If an annotation is already added to the split image it is skipped.
    If an overlapping annotation is already added to another split image, it is noted
    to later move this annotation to the new image.

    Args:
        - already_loaded_ids_image: List[int] = list of annotation ids that already
            have been added to another split image
        - already_loaded_ids_flake:  List[int] = list of annotation ids that already
            have been added to the split image
        - image_info: dict = all information of the original image, contains annotations
        - bbox: Polygon = bbox to use for overlap checks

    Returns:
        - list of new found overlapping annotations
        - list of ids of the new found overlapping annotations
        - list of overlapping ids from other split images
    """
    new_overlapping_annotations = []
    new_overlapping_ids = []
    split_overlapping_ids = []

    for annotation in image_info['annotations']:
        if annotation['id'] in already_loaded_ids_flake:
            continue

        if check_overlap(annotation, bbox):
            if annotation['id'] in already_loaded_ids_image:
                split_overlapping_ids.append(annotation['id'])
    
            new_overlapping_ids.append(annotation['id'])
            new_overlapping_annotations.append(annotation)
    
    if len(new_overlapping_ids) != 0:
        print('{} new overlapping annotations found: {}'.format(len(new_overlapping_ids), new_overlapping_ids))
    else:
        print('No new overlapping annotations found')
    return new_overlapping_annotations, new_overlapping_ids, split_overlapping_ids

def add_overlapping_annotations(
    split_id: int,
    overlapping_annotations: List[dict],
    filename: str,
):
    """
    Function to create annotations from the overlapping annotations for 
    the global annotations dictionary. 

    Args:
        - split_id: int = id of the split image
        - overlapping_annotations: List[dict] = all overlapping annotations
        - filename: str = filename of the corresponding split image

    Returns:
        - list of all bbox coordinates
        - list of all updated annotations
    """
    bbox_coords = []
    annotations = []

    for annotation in overlapping_annotations:
        bbox_coords.append(bbox_to_coords(annotation['bbox']))

        flake_annotation = annotation.copy()
        flake_annotation['image_id'] = filename
        flake_annotation['id'] = int('{}{}'.format(split_id, flake_annotation['id']))
        
        annotations.append(flake_annotation)

    return bbox_coords, annotations

def reset_image_dir(path: str) -> None:
    """Function to reset the /batchsplit directory."""
    print(f'Reseting {path} directory')
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    
    return None

def change_split_image(ids_to_switch: List[int], new_split_image: str, annotations_dict: dict):
    """
    Function to move annotations from one split image to another.

    Args:
        - ids_to_switch: List[int] = annotation ids that need to be moved
        - new_split_image: str = the split image to which the annotation ids will be moved
        - annotations_dict: dict = the global annotations dictionary that will be updated

    Returns:
        - updated global annotations dictionary
        - list of all swichted bboxs
    """
    print('\nChanging split image of annotations:\n{}'.format(ids_to_switch))
    bbox_coords = []

    for annotation in annotations_dict['annotations']:
        split_id = annotation['image_id'].split('split_')[-1].split('.')[0]
        
        for id_to_switch in ids_to_switch:
            updated_id = int('{}{}'.format(split_id, id_to_switch))
    
            if annotation['id'] == updated_id:
                print("[id: {}, updated id: {}] Changing {} to {}".format(id_to_switch, updated_id, annotation['image_id'], new_split_image))
                annotation['image_id'] = new_split_image
                bbox_coords.append(bbox_to_coords(annotation['bbox']))
    
    return annotations_dict, bbox_coords

def delete_zero_annotation_images(ROOT_DIR: str, annotations_dict: dict):
    """
    Function that deletes all split images with zero annotations.
    Images with zero annotation are created after annotations are moved to
    other split images.
    """
    for split_image in annotations_dict['images']:
        annotation_count = sum(1 for i in annotations_dict['annotations'] if i['image_id'] == split_image['file_name'])
        
        if annotation_count == 0:
            print('Deleting split image: {}'.format(split_image['file_name']))
            os.remove(os.path.join(ROOT_DIR, 'data', 'images', 'batchsplit', split_image['file_name']))
            annotations_dict['images'].remove(split_image)
    
    return annotations_dict

def extract_bbox_coords(bbox_coords: list):
    """
    Function to extract the smallest and largest x and y coordinates
    from all bbox coordinates.
    """
    x_coords = [[i[0], i[2]] for i in bbox_coords]
    x_coords = [i for j in x_coords for i in j]
    
    y_coords = [[i[1], i[3]] for i in bbox_coords]
    y_coords = [i for j in y_coords for i in j]

    updated_coords = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    return updated_coords