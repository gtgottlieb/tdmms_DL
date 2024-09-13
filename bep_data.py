"""Module that creates a custom dataset to load custom data"""

import os
import json
import cv2
import numpy as np

from pycocotools.coco import COCO
from mrcnn.utils import Dataset
from pycocotools import mask as maskUtils


class bepDataset(Dataset):
    """
    Class to load the BEP data.
    """
    class_variable_mapping = {
        'mono': 1,
        'few': 2,
        'thick': 3
    }
            
    def convert_annotations_to_coco(self, annotations_file: str, img_dir: str):
        """
        Function to convert the Labelbox.com .ndjson annotation output 
        to the classic COCO annotation format.
        
        Args:
            - annotations_file: str, path to .ndjson annotation file, WITHOUT file extension
                example: '../../test_annotations'
            - img_dir: str, path to image directory
        Returns:
            - COCO format dictionary
        """
        with open(annotations_file+'.ndjson') as f:
            rows = [json.loads(l) for l in f.readlines()]
            
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
                {"id": 1, "name": "Mono_Graphene"},
                {"id": 2, "name": "Few_Graphene"},
                {"id": 3, "name": "Thick_Graphene"}
            ],
            "annotations": [],
            "images": [],
        }

        annotation_id = 1
        
        for row in rows:
            for label in list(row['projects'].values())[0]['labels']:
                for obj in label['annotations']['objects']:
                    segmentation = []
                    for point in obj["polygon"]:
                        segmentation += [point["x"], point["y"]]
                    
                    x = [point["x"] for point in obj["polygon"]]
                    y = [point["y"] for point in obj["polygon"]]
                    bbox = [min(x), min(y), max(x) - min(x), max(y) - min(y)]
                    
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": row['data_row']['external_id'],
                        "category_id": self.class_variable_mapping[obj['value']],
                        "segmentation": [segmentation],
                        "area": bbox[2] * bbox[3],
                        "bbox": bbox,
                        "iscrowd": 0,
                        "source": "ali"
                    }
                    coco_format['annotations'].append(annotation_info)
                    annotation_id += 1
                    
        imgs = [i for i in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, i))]
        
        img_sample = cv2.imread(os.path.join(img_dir, imgs[0]))
        img_height, img_width, _ = img_sample.shape
        
        for idx, img in enumerate(imgs):
            coco_format['images'].append({
                "id": idx,
                "path": os.path.join(img_dir, img),
                "file_name": img,
                "width": img_width,
                "height": img_height,
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            })
                    
        with open(annotations_file+'.json', 'w+') as f:
            json.dump(coco_format, f)
        
        return None
    
    def load_dir(self, path: str, data_type: str, reload_annotations=False):
        """"Function to load image directory"""
        img_dir = os.path.join(path, data_type)
        annotations_file = os.path.join(path, 'annotations', f'{data_type}')
        
        if not os.path.isfile(annotations_file+'.json') or reload_annotations:
            self.convert_annotations_to_coco(annotations_file, img_dir)
                        
        coco = COCO(annotations_file + '.json')
        
        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("ali", i, coco.loadCats(i)[0]["name"])
                
        for i in image_ids:
            self.add_image(
                "ali",
                image_id=i,
                path=os.path.join(img_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[coco.imgs[i]['file_name']],
                    catIds=class_ids,
                    iscrowd=None
                )))
    
    def load_mask(self, image_id):
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
            class_id = self.map_source_class_id(
                "ali.{}".format(annotation['category_id']))
            
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
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
        
    def __str__(self):
        return '\n'.join([i['path'] for i in self.image_info])