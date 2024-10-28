"""Module for the dataset used for MAL."""

import os
import sys

sys.path.append(os.path.abspath('../'))

from mrcnn import utils

class malDataset(utils.Dataset):
    """
    Class to load data for MAL. Used for prelabeling
    unlabeled batches.
    
    Data should be stored in the following way:
    
    ROOT_DIR/
        data/
            annotations/ (.ndjson or .json)
                batch<i>.ndjson
            images/
                batch<i>/
                    [image_1_name].png
                    [image_2_name].png
                    .
                    .
    """
    
    class_variable_mapping = {
        'mono': 1,  # 1 layer           / 0.7 nm
        'few': 2,   # 2 - 10 layers     / 1.4 - 7 nm 
        'thick': 3  # 10 - 40 layers    / 7 - 28 nm
    }
     
    def __init__(self):
        super().__init__()
        
        self.image_id = 1        
            
    def load_dir(
        self,
        path: str, 
        data: str,
    ):
        """"
        Function to load image directory.
        
        Args:
            - path: path to where the images/ and annotations/ folders are located
            - data: data type, such as: 'train', 'val', 'bacth1', 'batch2', etc. 
        """
        self.path = path
        self.data = data

        data_dir = os.path.join(path, 'images', data)
    
        images = os.listdir(data_dir)
        
                
        for i in images:
            self.add_image(
                "ali",
                image_id=self.image_id,
                path=os.path.join(data_dir, i+'.png'),
            )
            self.image_id += 1
            
    def load_multiple_dir(self, path: str, dirs: list):
        """Function to load multiple directories, such as multiple batches.
        All directories are accessed by using the given path.
        
        Args:
            - path: path to where the images/ and annotations/ folders are located
            - dirs: list of directories to load, for example, ['batch1', 'batch2', 'batch3']
        """
        
        for dir in dirs:
            self.load_dir(path, dir)
        
        return None
        
    def __str__(self):
        s = 'Path: {}'.format(self.path)
        s += '\nData: {}'.format(self.data)
        # s += 'Images and annotations:\n' + '\n'.join([i for i in self.image_info])

        return s
    
if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("../../../")
    dataset = malDataset()
    dataset.load_dir(os.path.join(ROOT_DIR, 'data'), 'batch4')

    print(dataset.image_info)
