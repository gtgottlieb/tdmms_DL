"""Module that creates a custom dataset to load custom data"""

import os
import json

from pycocotools.coco import COCO
from mrcnn.utils import Dataset

class bepDataset(Dataset):
    """
    Class to load the BEP data.
    """
    class_variable_mapping = {
        'mono': 1,
        'few': 2,
        'thick': 3
    }
            
    def convert_annotations_to_coco(self, path: str):
        """
        Function to convert the Labelbox.com .ndjson annotation output 
        to the classic COCO annotation format.
        
        Args:
            - path: str, path to .ndjson annotation file, WITHOUT file extension
                example: '../../test_annotations'
        Returns:
            - COCO format dictionary
        """
        with open(path+'.ndjson') as f:
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
            "annotations": [],
            "categories": ['mono', 'few', 'thick']
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
                    
        with open(path+'.json', 'w+') as f:
            json.dump(coco_format, f)
        
        return None
    
    def load_dir(self, path: str, data_type: str, reload_annotations=False):
        """"Function to load image directory"""
        img_dir = os.path.join(path, data_type)
        annotations_file = os.path.join(path, 'annotations', f'{data_type}')
        
        if not os.path.isfile(annotations_file+'.json') or reload_annotations:
            self.convert_annotations_to_coco(annotations_file)
        
        annotations_file += '.json'
        
        imgs = [i for i in img_dir if os.path.isfile(os.path.join(img_dir, i))]        
        coco = COCO(annotations_file)
        
        class_ids = sorted(coco.getCatIds())
        
        print(class_ids)

        
        for i in imgs:
            self.add_image(
                path=os.path.join(path, i),
                source=path,
                image_id=i
            )
        
    def __str__(self):
        return '\n'.join([i['path'] for i in self.image_info])