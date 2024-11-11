"""Module to Slit an image with multiple flakes / segmentations into multiple images"""

"""
Things that do not work yet:
    - Annotations get linked to the wrong split image
    - Annotations disappear
    - Split image cuts of part of flake, annotation is added but not the entire bounding box
        is in the splitflake
    - Flake is missing

Idea for development:
    - Add a box in the image of the final bbox instead of creating an image split
"""

import cv2
import os
import sys
import json
import shutil
import contextlib
import numpy as np

from typing import Tuple, List
from shapely.geometry import box, Polygon, Point

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))

from bep.utils import load_train_val_datasets

def create_background(
    square_size: int = 100,
    sample_image_path: str = os.path.join(ROOT_DIR, 'data', 'images', 'batch4', '67_sio2_NbSe2_Exfoliation_C5-84_f4_img.png')
) -> Tuple[np.ndarray, int, int]:
    sample_image = cv2.imread(sample_image_path)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    small_square = cv2.GaussianBlur(sample_image[:square_size, :square_size], (15,15), 0)

    tile_height, tile_width = small_square.shape[:2]
    num_tiles_y = sample_image.shape[0] // tile_height + 1
    num_tiles_x = sample_image.shape[1] // tile_width + 1
    tiled_background = np.tile(small_square, (num_tiles_y, num_tiles_x, 1))[:sample_image.shape[0], :sample_image.shape[1], :]
    tiled_background = cv2.GaussianBlur(tiled_background, (11, 11), 0)

    return tiled_background, sample_image.shape[1], sample_image.shape[0]

def get_images_to_split(image_info: List[dict], annotation_threshold: int) -> list:
    images = []
    
    for i in image_info:
        annotation_count = len(i['annotations'])

        if annotation_count >= annotation_threshold:
            images.append(i['path'].split('\\')[-1])

    return images

def get_image_ids_positions(image_info: List[dict], images: List[str]) -> list:
    all_image_ids = [i['id'] for i in image_info]
    image_ids = [i['id'] for i in image_info if any(j in i['path'] for j in images)]
    image_id_positions = [all_image_ids.index(i) for i in image_ids]

    return image_id_positions

def update_annotations_dict(
    filename: str,
    annotations_dict: dict,
    idx: int,
    annotation: dict,
    image_id: int,
    width: int,
    height: int,
    bbox: List[float]
) -> dict:

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
    filename: str,
    flake_image: np.ndarray,
) -> None:
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
    x, y, w, h = [int(i) for i in bbox]
    borders = np.array([y-border, y+h+border, x-border, x+w+border]).clip(0)

    flake_image = tiled_background.copy()
    flake_image[borders[0]:borders[1], borders[2]:borders[3]] = image[borders[0]:borders[1], borders[2]:borders[3]]

    return flake_image

def bbox_to_coords(bbox):
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)

def coords_to_bbox(coords):
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
    split_id,
    overlapping_annotations: List[dict],
    filename: str,
):
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
    print(f'Reseting {path} directory')
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    
    return None

def change_split_image(ids_to_switch, new_split_image, annotations_dict):
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

def delete_zero_annotation_images(annotations_dict):
    for split_image in annotations_dict['images']:
        annotation_count = sum(1 for i in annotations_dict['annotations'] if i['image_id'] == split_image['file_name'])
        
        if annotation_count == 0:
            print('Deleting split image: {}'.format(split_image['file_name']))
            os.remove(os.path.join(ROOT_DIR, 'data', 'images', 'batchsplit', split_image['file_name']))
            annotations_dict['images'].remove(split_image)
    
    return annotations_dict

def extract_bbox_coords(bbox_coords: list):
    x_coords = [[i[0], i[2]] for i in bbox_coords]
    x_coords = [i for j in x_coords for i in j]
    
    y_coords = [[i[1], i[3]] for i in bbox_coords]
    y_coords = [i for j in y_coords for i in j]

    updated_coords = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    return updated_coords

def split_images(
    annotation_threshold: int = 15,
    border: int = 10,
    dataset: str = 'train',
    area_threshold: int = None,
) -> None:
    # Reset the /batchsplit image directory
    reset_image_dir(os.path.join(ROOT_DIR, 'data', 'images', 'batchsplit'))

    if dataset == 'train':
        data, _, _ = load_train_val_datasets(ROOT_DIR)

    # Get all the image filenames that contain more annotations
    # than the annotation_threshold
    images = get_images_to_split(data.image_info, annotation_threshold)

    # Create a background from a small part of backgorund without flakes
    tiled_background, width, height = create_background()

    # Get the positions of the image filenames in the image_info list
    image_id_positions = get_image_ids_positions(data.image_info, images)

    print(f'Images: {images}')

    annotations_file = os.path.join(ROOT_DIR, 'data', 'annotations', 'batchsplit.json')
    annotations_dict = {
        "info": {
            "year": 2024,
            "description": "",
            "version": 1.0,
            "date_created": "",
            "url": "tudelft.nl",
            "contributor": "Abel de Lange"
        }, 
        "categories": [
            {"id": 1, "name": "Mono_NbSe2"},
            {"id": 2, "name": "Few_NbSe2"},
            {"id": 3, "name": "Thick_NbSe2"}
        ],
        "annotations": [],
        "images": []
    }

    # Loop through images
    for image_id in image_id_positions:
        image_info = data.image_info[image_id]

        print('\nSplitting: {}'.format(image_info['path']))
        image = cv2.imread(image_info['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # List to track which annotations have been linked to a split image
        already_added_ids_image = []
        
        # Loop through every annotation in the image
        for annotation in image_info['annotations']:
            annotation_id = annotation['id']

            # Check if annotation is already looped through or added to a
            # split image
            if annotation_id in already_added_ids_image:
                print(f'Annotation {annotation_id} already in an image due to earlier overlap, skipping')
                continue
            already_added_ids_image.append(annotation_id)
            
            # See the first annotation as the parent annotation / flake
            # then overlapping annotations will be searched. Eventually when no
            # overlapping annotations are found a split image is created and stored.
            
            # This list tracks which annotation ids have been added to the parent
            # annotation
            already_loaded_ids_flake = [annotation_id] # original ids, single number, from data.image_info

            first_overlap_check = True # To enable the first overlap check
            last_overlap_count = 0 # Last amount of found overlaps, whether to check again

            print(f'\nAnalysing annotation with id {annotation_id}')

            bbox = tuple(annotation['bbox'])
            print(f'Inital bbox: {bbox}')
            x, y, w, h = (int(i) for i in bbox)

            filename = data.image_info[image_id]['path'].split('\\')[-1].split('.')[0] + f'_split_{annotation_id}' + '.png'

            # Lists to store all the x and y coordinates of the initial annotation
            # , all overlapping ones and the switches annotations.
            bbox_coords_list = [bbox_to_coords(bbox)]

            # Check for overlapping annotations untill none are found
            while first_overlap_check or last_overlap_count > 0:
                # flake_image = cut_out_flake(tiled_background, bbox, image, border)
                # filename_it = data.image_info[image_id]['path'].split('\\')[-1].split('.')[0] + f'_split_{annotation_id}' + '_bbox_' + '_'.join([str(i) for i in list(bbox)])+ '.png'
                # coords = bbox_to_coords(bbox)
                # coords = [int(i) for i in coords]
                # cv2.rectangle(flake_image, (coords[0], coords[1]), (coords[2], coords[3]), color=1, thickness=2)
                # store_image(filename_it, flake_image)

                if first_overlap_check:
                    # Use the inital bbox
                    print('\nRunning first overlap check..')
                    cut_out_bbox = box(x-border, y-border, x+w+border, y+h+border)
                else:
                    # Use the updated bbox
                    print('Running another overlap check..')
                    x, y, w, h = [int(i) for i in updated_bbox]
                    cut_out_bbox = box(x-border, y-border, x+w+border, y+h+border)

                # Check for any overlap between the bbox and any annotation polygon
                overlapping_annotations, overlapping_ids, split_overlapping_ids = check_overlap_image(
                    already_added_ids_image,
                    already_loaded_ids_flake,
                    image_info,
                    cut_out_bbox
                )

                if len(overlapping_ids) > 0:
                    already_added_ids_image += overlapping_ids
                    already_loaded_ids_flake += overlapping_ids
                    bbox_coords, updated_annotations = add_overlapping_annotations(
                        annotation_id,
                        overlapping_annotations,
                        filename,
                    )

                    print(f'Overlap bboxs: {bbox_coords}')

                    annotations_dict['annotations'] += updated_annotations
                    bbox_coords_list += bbox_coords

                    # coords = bbox_to_coords(bbox)
                    # coords = [int(i) for i in coords]
                    # cv2.rectangle(flake_image, (coords[0], coords[1]), (coords[2], coords[3]), color=5, thickness=2)
                
                if split_overlapping_ids:
                    annotations_dict, bbox_coords = change_split_image(split_overlapping_ids, filename, annotations_dict)
                    bbox_coords_list += bbox_coords

                    print(f'Split overlap bboxs: {bbox_coords}')
         
                updated_bbox = coords_to_bbox(extract_bbox_coords(bbox_coords_list))
                bbox = updated_bbox

                last_overlap_count = len(overlapping_ids)
                first_overlap_check = False

            annotations_dict = update_annotations_dict(
                filename,
                annotations_dict,
                annotation_id,
                annotation,
                image_id,
                width,
                height,
                bbox
            )

            print(f'Final bbox: {bbox}')
            print(f'Creating and storing image {filename}')

            flake_image = cut_out_flake(tiled_background, bbox, image, border)
            store_image(filename, flake_image)

    print('\nDeleting images with zero annotations')
    annotations_dict = delete_zero_annotation_images(annotations_dict)

    print('Storing annotations file')
    with open(annotations_file, 'w+') as f:
        json.dump(annotations_dict, f)
    
    return None

class NullWriter:
    def write(self, message):
        pass

if __name__ == '__main__':
    with contextlib.redirect_stdout(NullWriter):
        split_images(border=15)
    # split_images(border=15)
