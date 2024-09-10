"""Module that creates a custom dataset to load custom data"""

import os
import json

from mrcnn.utils import Dataset

class bepDataset(Dataset):
    class_variable_mapping = {
        'mono': 1,
        'few': 2,
        'thick': 3
    }
    
    def load_dir(self, path: str):
        """"Function to load image directory"""
        imgs = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))]
        
        for i in imgs:
            self.add_image(
                path=os.path.join(path, i),
                source=path,
                image_id=i
            )
            
    def convert_annotations_to_coco(self, path: str):
        """
        Function to convert the Labelbox.com .ndjson annotation output 
        to the classic COCO annotation format.
        
        Args:
            - path: str, path to .ndjson annotation file
        Returns:
            - COCO format dictionary
        """
        with open(path) as f:
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
            "annotations": []
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
        return coco_format
        
    def __str__(self):
        return '\n'.join([i['path'] for i in self.image_info])