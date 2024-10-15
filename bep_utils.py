"""Module that contains used functions."""

import json
import os
import random
import shutil

from bep_data import bepDataset
from tdmcoco import CocoDataset

data_types = ['images', 'annotations']
data_sets = ['train', 'val']

""""
Tensorflow logging levels:

0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed

Set by running: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""

def load_train_val_datasets(ROOT_DIR: str):
    dataset_train = bepDataset()
    dataset_train.load_dir(os.path.join(ROOT_DIR, 'data'), 'train', reload_annotations=True)
    dataset_train.prepare()

    dataset_val = bepDataset()
    dataset_val.load_dir(os.path.join(ROOT_DIR, 'data'), 'val', reload_annotations=True)
    dataset_val.prepare()

    return dataset_train, dataset_val

def check_dir_setup(ROOT_DIR: str, train_size: float):
    """Function to check if the directory is setup correctly.
    Like:
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
    """  
    for dt in data_types:
        for ds in data_sets:
            extension = ''
            if dt == data_types[1]:
                extension = '.ndjson'
            
            path = os.path.join(ROOT_DIR, 'data', dt, ds+extension)
            if not os.path.exists(path):
                print(f'{path} did not exist')
                create_dir_setup(ROOT_DIR, train_size)
                return None
            
    print('Directory setup correctly')

def create_dir_setup(ROOT_DIR: str, train_size: float):
    """Function to reset and create train and validation directories."""
    
    print('Creating directories from batches..')

    batches = [i for i in os.listdir(os.path.join(ROOT_DIR, 'data', 'images')) if 'batch' in  i]
    print('Found batches:',', '.join(batches))
    
    reset_dirs(ROOT_DIR)
    data_split_images(batches, ROOT_DIR, train_size)
    data_split_annotations(batches, ROOT_DIR)
    
def reset_dirs(ROOT_DIR: str):
    """Function to reset the image and annotation directories of the
    train and validation sets."""
    # Reset image directory
    for ds in data_sets:
        path = os.path.join(ROOT_DIR, 'data', 'images', ds)
        
        if os.path.exists(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                print('Error in deleting image directory:',e)
        
        os.mkdir(os.path.join(ROOT_DIR, 'data', 'images', ds))
                
    # Reset annotations directory
    for ds in data_sets:
        path = os.path.join(ROOT_DIR, 'data', 'annotations', ds+'.ndjson')
        
        if os.path.exists(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                print('Error in deleting annotations file:',e)
                
    return None
                        
def data_split_images(batches: list, ROOT_DIR: str, train_size: float):
    """Function to load the images from all found batches and split the 
    images into a train and validation set."""
    imgs_batches = []
    
    for batch in batches:
        imgs_batches.append((batch, os.listdir(os.path.join(ROOT_DIR, 'data', 'images', batch))))    
    imgs_batches = [[(i[0], j) for j in i[1]] for i in imgs_batches]
    imgs_batches = [i for j in imgs_batches for i in j]
    
    random.shuffle(imgs_batches)
    
    img_count = len(imgs_batches)    
    print(f'Total image count: {img_count}')
    
    train_amount = round(train_size*img_count)
    
    print('Copying images..')
    for i in imgs_batches[:train_amount]:
        # i = (batch_name, image_name)
        shutil.copy(
            os.path.join(ROOT_DIR, 'data', 'images', i[0], i[1]),
            os.path.join(ROOT_DIR, 'data', 'images', 'train')
        )
    
    for i in imgs_batches[train_amount:]:
        # i = (batch_name, image_name)
        shutil.copy(
            os.path.join(ROOT_DIR, 'data', 'images', i[0], i[1]),
            os.path.join(ROOT_DIR, 'data', 'images', 'val')
        )

    return None

def data_split_annotations(batches: list, ROOT_DIR: str):
    """Function to load the annotations from all found batches and split the 
    annotations into a train and validation set.
    
    TODO: 'f.write(str(row)+'\n')' writes the .ndjson lines as a string, containing
    ' instead of ", and ' is no .json. It work with a .replace when reading the files.
    But it can better be fixed here.
    """
    rows = []
    for batch in batches:
        with open(os.path.join(ROOT_DIR, 'data', 'annotations', batch+'.ndjson')) as f:
            rows += [json.loads(l) for l in f.readlines()]
    
    train_imgs = os.listdir(os.path.join(ROOT_DIR, 'data', 'images', 'train'))
    val_imgs = os.listdir(os.path.join(ROOT_DIR, 'data', 'images', 'val'))
    
    print('Writing annotation files..')
    with open(os.path.join(ROOT_DIR, 'data', 'annotations', 'train.ndjson'), "w+") as f:
        for row in rows:
            if row['data_row']['external_id'] in train_imgs:
                f.write(str(row)+'\n')
                
    with open(os.path.join(ROOT_DIR, 'data', 'annotations', 'val.ndjson'), "w+") as f:
        for row in rows:
            if row['data_row']['external_id'] in val_imgs:
                f.write(str(row)+'\n')
                
if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("../")
    check_dir_setup(ROOT_DIR, 0.7)
    # create_dir_setup(ROOT_DIR, 0.7)