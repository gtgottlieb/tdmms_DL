"""Module that contains used functions."""

import json
import os
import random
import shutil
import matplotlib.pyplot as plt

from bep_data import bepDataset
from tdmcoco import CocoDataset

from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.model import log

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

def load_train_val_datasets_tdmms(ROOT_DIR: str, material: str = 'MoS2'):
    dataset_train = CocoDataset()
    dataset_train.load_coco(os.path.join(ROOT_DIR, 'DL_2DMaterials', 'Dataset_DL_2DMaterials', material), 'train')
    dataset_train.prepare()

    dataset_val = CocoDataset()
    dataset_val.load_coco(os.path.join(ROOT_DIR, 'DL_2DMaterials', 'Dataset_DL_2DMaterials', material), 'val')
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

def get_ax(rows=1, cols=1, size=15):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class runModel():
    def __init__(self, model, config, dataset=None):
        self.model = model
        self.config = config
        self.image_id = None
        self.dataset = dataset
    
    def run(self, dataset=None, rand=False, image_idx=0):
        """
        Function to run the model on an image.
        
        Args:
            - dataset: if a dataset here is provided it overwrites a previously loaded dataset
                        in the object.
            - rand: True or False, if a random image is chosen from the dataset or not
            - image_idx: choose the image index from the dataset to use

        """
        if dataset:
            self.dataset = dataset
        assert self.dataset

        self.image_id = self.dataset.image_ids[image_idx]
        if rand:
            self.image_id = random.choice(self.dataset.image_ids)
            
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(self.dataset, self.config, self.image_id)

        info = self.dataset.image_info[self.image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], self.image_id, self.dataset.image_reference(self.image_id)))

        results = self.model.detect([image], verbose=1)

        title = self.model.name + ' Predictions'

        # Display results
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(
            image, r['rois'],
            r['masks'],
            r['class_ids'], 
            ['','Mono', 'Few','Thick'], r['scores'],
            ax=ax,
            title=title)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
    
    def gt(self, dataset=None, rand=False, image_idx=0):
        """Function to show the ground truth of the image, on which run() made predictions."""
        if dataset:
            self.dataset = dataset
        assert self.dataset

        if not self.image_id and self.image_id != 0:
            print('Please run .run() befor .gt().')
            raise AssertionError

        image = self.dataset.load_image(self.image_id)
        mask, class_ids = self.dataset.load_mask(self.image_id)
        original_shape = image.shape
        # Resize
        image, window, scale, padding, _ = utils.resize_image(
            image, 
            min_dim=self.config.IMAGE_MIN_DIM, 
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE)
        mask = utils.resize_mask(mask, scale, padding)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)

        # Display image and additional stats
        print("image_id: ", self.image_id, self.dataset.image_reference(self.image_id))
        print("Original shape: ", original_shape)
        log("image", image)
        log("mask", mask)
        log("class_ids", class_ids)
        print(class_ids)
        log("bbox", bbox)
        # Display image and instances
        ax = get_ax(1)
        title = 'Ground Truth'
        visualize.display_instances(
            image,
            bbox,
            mask,
            class_ids,
            self.dataset.class_names,
            ax=ax, 
            title=title
        )
                
if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("../")
    # check_dir_setup(ROOT_DIR, 0.7)
    create_dir_setup(ROOT_DIR, 0.7)