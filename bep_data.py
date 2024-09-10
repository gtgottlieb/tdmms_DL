"""Module that creates a custom dataset to load custom data"""

import os

from mrcnn.utils import Dataset

class bepDataset(Dataset):
    def load_dir(self, path: str):
        imgs = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))]
        
        for idx, i in enumerate(imgs):
            self.add_image(
                id=i,
                path=os.path.join(path, i),
                source=path,
                image_id=idx
            )