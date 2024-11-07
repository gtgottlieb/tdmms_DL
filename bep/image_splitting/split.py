"""Module to Slit an image with multiple flakes / segmentations into multiple images"""

"""
Things that do not work yet:
    - Annotations get linked to the wrong split image
    - Annotations disappear
    - Often a very small part of a flake appears in the split image
        of another flake
    - Many edge cases when there are many flakes
        - Small split image appears in multiple largers split images
    - Code is probably unnecessarily complicated
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

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))

from bep.utils import load_train_val_datasets
from tdmms.tdmcoco import CocoDataset

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

def get_images_to_split(dataset: CocoDataset, annotation_threshold: int) -> list:
    images = []
    
    for i in dataset.image_info:
        annotation_count = len(i['annotations'])

        if annotation_count >= annotation_threshold:
            images.append(i['path'].split('\\')[-1])

    return images

def get_image_ids_positions(dataset: CocoDataset, images: List[str]) -> list:
    all_image_ids = [i['id'] for i in dataset.image_info]
    image_ids = [i['id'] for i in dataset.image_info if any(j in i['path'] for j in images)]
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
    bbox,
    image: np.ndarray,
    border: int,
):
    x, y, w, h = [int(i) for i in bbox]

    flake_image = tiled_background.copy()
    flake_image[y-border:y+h+border, x-border:x+w+border] = image[y-border:y+h+border, x-border:x+w+border]

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
    already_switched_ids_flake: List[int],
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
            if (annotation['id'] in already_loaded_ids_image) and (annotation['id'] not in already_switched_ids_flake):
                print('Overlap with a flake from another split {}'.format(annotation['id']))
                split_overlapping_ids.append(annotation['id'])
            else:
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
    
    x_coords = [[i[0], i[2]] for i in bbox_coords]
    x_coords = [i for j in x_coords for i in j]
    
    y_coords = [[i[1], i[3]] for i in bbox_coords]
    y_coords = [i for j in y_coords for i in j]

    return x_coords, y_coords, annotations

def reset_image_dir(path: str) -> None:
    print(f'Reseting {path} directory')
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    
    return None

def change_split_image(ids_to_switch, new_split_image, annotations_dict):
    print('\nChanging split image of annotations:\n{}'.format(ids_to_switch))
    switched_ids = []

    for annotation in annotations_dict['annotations']:
        split_id = annotation['image_id'].split('split_')[-1].split('.')[0]
        for id_to_switch in ids_to_switch:
            updated_id = int('{}{}'.format(split_id, id_to_switch))
            if annotation['id'] == updated_id:
                print("[{}, {}] Changing {} to {}".format(id_to_switch, updated_id, annotation['image_id'], new_split_image))
                annotation['image_id'] = new_split_image
                switched_ids.append(id_to_switch)

    for split_image in annotations_dict['images']:
        annotation_count = sum(1 for i in annotations_dict['annotations'] if i['image_id'] == split_image['file_name'])
        # print('Split image: {}, annotation count: {}'.format(split_image['file_name'], annotation_count))
        if annotation_count == 0:
            print('Deleting split image: {}'.format(split_image['file_name']))
            os.remove(os.path.join(ROOT_DIR, 'data', 'images', 'batchsplit', split_image['file_name']))
            annotations_dict['images'].remove(split_image)
    
    return annotations_dict, switched_ids

def split_images(
    annotation_threshold: int = 15,
    border: int = 10,
    dataset: str = 'train',
    area_threshold: int = None,
) -> None:
    reset_image_dir(os.path.join(ROOT_DIR, 'data', 'images', 'batchsplit'))

    if dataset == 'train':
        data, _, _ = load_train_val_datasets(ROOT_DIR)

    images = get_images_to_split(data, annotation_threshold)

    tiled_background, width, height = create_background()
    image_id_positions = get_image_ids_positions(data, images)

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

    for image_id in image_id_positions:
        print('\nSplitting: {}'.format(data.image_info[image_id]['path']))
        image = cv2.imread(data.image_info[image_id]['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_info = data.image_info[image_id]

        already_added_ids_image = []
        
        for annotation in image_info['annotations']:
            annotation_id = annotation['id']
            if annotation_id in already_added_ids_image:
                print(f'Annotation {annotation_id} already in an image due to earlier overlap, skipping')
                continue
            already_added_ids_image.append(annotation_id)

            already_added_ids_flake = [annotation_id]
            already_switched_ids_flake = []

            first_overlap_check = True
            last_overlap_count = 0

            print(f'\nAnalysing annotation with id {annotation_id}')

            bbox = annotation['bbox']
            print(f'Inital bbox: {bbox}')
            x, y, w, h = (int(i) for i in bbox)

            filename = data.image_info[image_id]['path'].split('\\')[-1].split('.')[0] + f'_split_{annotation_id}' + '.png'

            # filename_initial = data.image_info[image_id]['path'].split('\\')[-1].split('.')[0] + f'_split_{annotation_id}_inital_bbox' + '.png'
            # flake_image = cut_out_flake(tiled_background, bbox, image, border)
            # store_image(filename_initial, flake_image)

            if area_threshold:
                area = w*h
                if area < area_threshold:
                    continue

            x_coords_list = []
            y_coords_list = []

            while first_overlap_check or last_overlap_count > 0:
                if first_overlap_check:
                    print('\nRunning first overlap check..')
                    cut_out_bbox = box(x-border, y-border, x+w+border, y+h+border)
                else:
                    print('Running another overlap check..')
                    x, y, w, h = [int(i) for i in updated_bbox]
                    cut_out_bbox = box(x-border, y-border, x+w+border, y+h+border)

                overlapping_annotations, overlapping_ids, split_overlapping_ids = check_overlap_image(
                    already_added_ids_image,
                    already_added_ids_flake,
                    already_switched_ids_flake,
                    image_info,
                    cut_out_bbox
                )

                if len(overlapping_ids) != 0:
                    already_added_ids_image += overlapping_ids
                    already_added_ids_flake += overlapping_ids
                    x_coords, y_coords, updated_annotations = add_overlapping_annotations(
                        annotation_id,
                        overlapping_annotations,
                        filename,
                    )

                    annotations_dict['annotations'] += updated_annotations
                    x_coords_list += x_coords
                    y_coords_list += y_coords

                    updated_coords = (min(x_coords_list), min(y_coords_list), max(x_coords_list), max(y_coords_list))
                    updated_bbox = coords_to_bbox(updated_coords)
                    bbox = updated_bbox
                
                if split_overlapping_ids:
                    annotations_dict, switched_ids = change_split_image(split_overlapping_ids, filename, annotations_dict)
                    already_switched_ids_flake += switched_ids

                last_overlap_count = len(overlapping_ids)
                first_overlap_check = False

            print(f'Final bbox: {bbox}')
            print('Creating and storing image')

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

            flake_image = cut_out_flake(tiled_background, bbox, image, border)
            store_image(filename, flake_image)

    print('Storing annotations file')
    with open(annotations_file, 'w+') as f:
        json.dump(annotations_dict, f)
    
    return None

class NullWriter:
    def write(self, message):
        pass

if __name__ == '__main__':
    # with contextlib.redirect_stdout(NullWriter)):
        # split_images(border=20)
    split_images(border=40)
