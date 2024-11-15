"""Module that creates a custom dataset to load custom data"""

import os
import sys
import contextlib
import json
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from pycocotools.coco import COCO
from mrcnn.utils import Dataset
from pycocotools import mask as maskUtils

class NullWriter:
    def write(self, message):
        pass

class bepDataset(Dataset):
    """
    Class to load the BEP data.
    
    Data should be stored in the following way:
    
    ROOT_DIR/
        data/
            annotations/ (.ndjson or .json)
                batch1.ndjson
                batch2.ndjson
                .
                .
                train.ndjson
                val.ndjson
            images/
                batch1/
                    [image_1_name].png
                    [image_2_name].png
                    .
                    .
                batch2/
                .
                .
                train/
                val/
            
    Use the check_dir_setup() function from bep.utils.py to
    check if the directory is setup correctly and create a 
    train validation split from the batches.
    """
    
    class_variable_mapping = {
        'mono': 1,  # 1 layer           / 0.7 nm
        'few': 2,   # 2 - 10 layers     / 1.4 - 7 nm 
        'thick': 3  # 10 - 40 layers    / 7 - 28 nm
    }
     
    def __init__(self):
        super().__init__()
        
        self.annotation_id = 1
        self.image_id = 1        
            
    def convert_annotations_to_coco(self, annotations_file: str, img_dir: str) -> None:
        """
        Function to convert the Labelbox.com .ndjson annotation output 
        to the classic COCO annotation format in .json. The .json file
        will be created next to the .ndjson file.
        
        Args:
            - annotations_file: str, path to .ndjson annotation file, WITHOUT file extension
                example: '../../test_annotations'
            - img_dir: str, path to image directory
        """
        with open(annotations_file+'.ndjson') as f:
            rows = [json.loads(l.replace('\'', '\"')) for l in f.readlines()]
            
        coco_format = {
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
            "images": [],
        }
        
        for row in rows:
            for label in list(row['projects'].values())[0]['labels']:
                for obj in label['annotations']['objects']:                        
                    segmentation = self.generate_segmentation(obj['polygon'])
                    bbox = self.generate_bbox(obj['polygon'])
                    
                    annotation_info = {
                        "id": self.annotation_id,
                        "image_id": row['data_row']['external_id'],
                        "category_id": self.class_variable_mapping[obj['value']],
                        "segmentation": [segmentation],
                        "area": bbox[2] * bbox[3],
                        "bbox": bbox,
                        "iscrowd": 0,
                        "source": 'ali'
                    }
                    coco_format['annotations'].append(annotation_info)
                    self.annotation_id += 1
                    
        imgs, img_height, img_width = self.get_image_info(img_dir)
        
        for img in imgs:
            coco_format['images'].append({
                "id": self.image_id,
                "path": os.path.join(img_dir, img),
                "file_name": img,
                "width": img_width,
                "height": img_height,
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            })
            self.image_id += 1
                    
        with open(annotations_file+'.json', 'w+') as f:
            json.dump(coco_format, f)
        
        return None
    
    def load_dir(
        self,
        path: str, 
        data: str,
        reload_annotations: bool = False,
        return_coco = False
    ):
        """"
        Function to load image directory
        
        Args:
            - path: path to where the images/ and annotations/ folders are located
            - data: data type, such as: 'train', 'val', 'bacth1', 'batch2', etc. 
            - reload_annotations: determines if the .json files will be reset
            - return_coco: determines if the COCO object will be returned or not
        """
        self.path = path
        self.data = data

        data_dir = os.path.join(path, 'images', data)
        annotations_file = os.path.join(path, 'annotations', data)
        
        if not os.path.isfile(annotations_file+'.json') or reload_annotations:
            self.convert_annotations_to_coco(annotations_file, data_dir)
        
        with contextlib.redirect_stdout(NullWriter()):
            coco = COCO(annotations_file + '.json')

        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("ali", i, coco.loadCats(i)[0]["name"])
                
        for i in image_ids:
            self.add_image(
                "ali",
                image_id=i,
                path=os.path.join(data_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[coco.imgs[i]['file_name']],
                    catIds=class_ids,
                    iscrowd=None
                )))
        
        if return_coco:
            return coco
        else:
            return None
            
    def load_multiple_dir(self, path: str, dirs: list, reload_annotations=False):
        """Function to load multiple directories, such as multiple batches.
        All directories are accessed by using the given path.
        
        Args:
            - path: path to where the images/ and annotations/ folders are located
            - dirs: list of directories to load, for example, ['batch1', 'batch2', 'batch3']
            - reload_annotations: determines if the .json files will be reset
        """
        
        for dir in dirs:
            self.load_dir(path, dir, reload_annotations)
        
        return None
    
    def load_split(self, path: str):
        if os.path.isdir(os.path.join(path, 'images', 'batchsplit')):
            split_images = list(set([i.split('_split')[0] for i in os.listdir(os.path.join(path, 'images', 'batchsplit'))]))        
            self.image_info = [i for i in self.image_info if not any(j in i['path'] for j in split_images)]

            self.load_dir(path, 'batchsplit')
        else:
            print('No batchsplit data found, please run bep/image_splitting/split.py first.\nSkipping split data')
        return None
    
    def load_mask(self, image_id, rotate: bool = False, shape: int = None):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            loop_annotation = annotation.copy()
            if rotate:
                print('Rotating annotation')
                loop_annotation = self.rotate_180_clockwise(annotation, shape[0], shape[1])
            

            class_id = self.map_source_class_id(
                "ali.{}".format(loop_annotation['category_id']))
            
            if class_id:
                m = self.annToMask(loop_annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if loop_annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super().load_mask(image_id)
    
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle
    
    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
    
    @staticmethod
    def generate_bbox(polygon: list):
        """Function to generate the bbox from a polygon object."""
        x = [point["x"] for point in polygon]
        y = [point["y"] for point in polygon]
        bbox = [min(x), min(y), max(x) - min(x), max(y) - min(y)]
        
        return bbox
    
    @staticmethod
    def generate_segmentation(polygon: list):
        """Function to generate a segmentation list in COCO format from a polygon object."""
        segmentation = []
        for point in polygon:
            segmentation += [point['x'], point['y']]
        
        return segmentation
    
    @staticmethod
    def get_image_info(img_dir: str):
        """Function to get all image files from directory and image
        width and height, assuming all the width and heights are the same.
        
        TODO: Get width and height for each image, but is maybe too slow and not worth it.
        """
        
        imgs = [i for i in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, i))]
        
        img_sample = cv2.imread(os.path.join(img_dir, imgs[0]))
        img_height, img_width, _ = img_sample.shape
        
        return imgs, img_height, img_width
    
    @staticmethod
    def rotate_clockwise(annotation, original_height):
        def rotate_polygon(polygon):
            rotated_polygon = []
            for i in range(0, len(polygon), 2):
                x, y = polygon[i], polygon[i + 1]
                new_x = y
                new_y = original_height - x
                rotated_polygon.extend([new_x, new_y])
            return rotated_polygon

        def rotate_bbox(bbox):
            x, y, width, height = bbox
            new_x = y
            new_y = original_height - (x + width)
            return [new_x, new_y, height, width]

        # Rotate segmentation
        annotation['segmentation'] = [
            rotate_polygon(polygon) for polygon in annotation['segmentation']
        ]

        # Rotate bbox
        annotation['bbox'] = rotate_bbox(annotation['bbox'])

        return annotation
    
    @staticmethod
    def rotate_180_clockwise(annotation, original_width, original_height):
        def rotate_polygon(polygon, width, height):
            rotated_polygon = []
            for i in range(0, len(polygon), 2):
                x, y = polygon[i], polygon[i + 1]
                # First rotation
                temp_x = y
                temp_y = height - x
                # Second rotation
                new_x = width - temp_x
                new_y = temp_y
                rotated_polygon.extend([new_x, new_y])
            return rotated_polygon

        def rotate_bbox(bbox, width, height):
            x, y, box_width, box_height = bbox
            # First rotation
            temp_x = y
            temp_y = height - (x + box_width)
            # Second rotation
            new_x = width - (temp_x + box_height)
            new_y = temp_y
            return [new_x, new_y, box_width, box_height]

        # First rotation (90 degrees), then second rotation (90 degrees)
        annotation['segmentation'] = [
            rotate_polygon(polygon, original_width, original_height)
            for polygon in annotation['segmentation']
        ]

        annotation['bbox'] = rotate_bbox(annotation['bbox'], original_width, original_height)

        return annotation

        
    def __str__(self):
        s = 'Path: {}'.format(self.path)
        s += '\nData: {}'.format(self.data)
        # s += 'Images and annotations:\n' + '\n'.join([i for i in self.image_info])

        return s

if __name__ == '__main__':
    train = bepDataset()
    train.load_dir(os.path.join(ROOT_DIR, 'data'), 'train', reload_annotations=False)
    train.load_split(os.path.join(ROOT_DIR, 'data'))
    train.prepare()