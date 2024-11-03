"""
Module to train the AI

How to run from a terminal:
    1. activate your environment
    2. run: $ py train.py 
        with optional arguments:   
            --reload_data_dir <True or False> 
            --starting_material <MoS2, WTe2, Graphene or BN>
"""

import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf

from imgaug import augmenters as iaa
from tdmms.tdmcoco import CocoConfig
from bep.utils import (
    check_dir_setup, 
    create_dir_setup,
    load_train_val_datasets
)

ROOT_DIR = os.path.abspath("../")
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                                       SETUP GPUS                                          #
#                                                                                           #
#-------------------------------------------------------------------------------------------#

GPUs = ['0','1']

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(GPUs)

physical_devices = tf.config.list_physical_devices('GPU')
for i in physical_devices:
    print(f'Found physical device{i}')

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                                  SETUP LOGS DIRECTORY                                     #
#                                                                                           #
#-------------------------------------------------------------------------------------------#

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'training')

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
    print(f"Folder '{DEFAULT_LOGS_DIR}' created.")

#-------------------------------------------------------------------------------------------#
#                                                                                           #
#                                        TRAINING                                           #
#                                                                                           #
#-------------------------------------------------------------------------------------------#

class TrainingConfig(CocoConfig):
    GPU_COUNT = len(GPUs)
    IMAGES_PER_GPU = 2

    def __init__(
        self,
        train_images: int,
        val_images: int,
        starting_material: str,
        intensity: int,
    ):
        super().__init__()
        batch_size = self.GPU_COUNT * self.IMAGES_PER_GPU
        total_image_count = train_images + val_images
        self.STEPS_PER_EPOCH = train_images / batch_size
        # Checkpoint name format:
        # <fine-tuned on>_<fine-tuned from>_<images in train and validation set>_<intensity>_<batch size>_<epoch amount>
        
        self.CHECKPOINT_NAME = f'nbse2_{starting_material.lower()}_{total_image_count}_{intensity}_{batch_size}_'

def train_model(
    reload_data_dir: bool = False,
    starting_material: str = 'MoS2',
    intensity: int = 4
):
    """
    Function to train MRCNN.

    The epoch step size is the amount of iterations per epoch:
        step size = amount of images / batch size 
        batch size = gpu count * images per gpu

    MRCNN automatically saves the best weights during training.

    Args:
        - computer: if the data directories should be reloaded
        - starting_material: Which weights will be used for fine-tuning on NbSe2.
                    MoS2, BN, Graphene or WTe2
        - intensity: 1, 2, 3 or 4. Determines the amount of training.
            1: Only network heads
            2: adds ResNet stage 4 and up
            3: adds all layers
            4: add all layers again with a lower learning rate
    
    Data directory should be setup as the following:
    ROOT_DIR/
        data/
            annotations/ (.ndjson or .json)
                train.ndjson
                val.ndjson
                test.ndjson
            images/
                train/
                val/
                test/
        weights/
            <material>_mask_rcnn_tdm_120.h5
    """

    if reload_data_dir:
        create_dir_setup(ROOT_DIR, (0.8, 0.1, 0.1))
    else:
        check_dir_setup(ROOT_DIR, (0.8, 0.1, 0.1))
    dataset_train, dataset_val, _ = load_train_val_datasets(ROOT_DIR)

    config = TrainingConfig(
        len(dataset_train.image_ids),
        len(dataset_val.image_ids),
        starting_material,
        intensity
    )
    config.display()

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = modellib.MaskRCNN(
            mode="training",
            config=config,
            model_dir=DEFAULT_LOGS_DIR
        )

        MODEL_PATH = os.path.join(ROOT_DIR, 'weights', starting_material.lower()+'_mask_rcnn_tdm_0120.h5')

        print("Loading weights ", MODEL_PATH)
        model.load_weights(MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]) # Model weights for coco model
        # model.load_weights(COCO_MODEL_PATH, by_name=True) # Model weights for a different model??

        augmentation = iaa.SomeOf((0, None), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-180,180)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.CropAndPad(percent=(-0.25, 0.25)),
            #iaa.Multiply((0.5, 1.5)),
            #iaa.GaussianBlur(sigma=(0.0, 0.5)),
            #iaa.AdditiveGaussianNoise(scale=(0, 0.15*255))
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(0, iaa.Multiply((0.5,1.5)))
            ),
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(1, iaa.Multiply((0.5,1.5)))
            ),
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(2, iaa.Multiply((0.5,1.5)))
            ),
            iaa.WithChannels(0, iaa.Multiply((0.5,1.5))),
            iaa.WithChannels(1, iaa.Multiply((0.5,1.5))),
            iaa.WithChannels(2, iaa.Multiply((0.5,1.5)))
        ])
        '''
        augmentation = iaa.SomeOf((0, 3), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                        iaa.Affine(rotate=180),
                        iaa.Affine(rotate=270)])
        ])
        '''

        if intensity >= 1:
            # Training - Stage 1
            print("Training network heads")
            model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',
                augmentation=augmentation,
            )

        if intensity >= 2:    
            # Training - Stage 2
            # Finetune layers from ResNet stage 4 and up
            print("Fine tune Resnet stage 4 and up")
            model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=60,
                layers='4+',
                augmentation=augmentation
            )
        
        if intensity >= 3:
            # Training - Stage 3
            # Fine tune all layers
            print("Fine tune all layers")
            model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE /10,
                epochs=90,
                layers='all',
                augmentation=augmentation
            )
        
        if intensity >= 4:
            print("Reduce LR and further tune all layers")
            model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE /100,
                epochs=120,
                layers='all',
                augmentation=augmentation
            )

    # model_version = 'NbSe2_weights_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # os.mkdir(os.path.join(ROOT_DIR, 'saved_weights', model_version))
    # config.write_txt(os.path.join(ROOT_DIR, 'saved_weights', model_version, 'config.txt'))
    # model.save_weights(os.path.join(ROOT_DIR, 'saved_weights', model_version, 'mask_rcnn_tdm_final.h5')) # Does not work

    return None   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train model'
    )

    parser.add_argument(
        '--reload_data_dir', 
        required=False,
        default=False,
        help='False or True'
    )

    parser.add_argument(
        '--starting_material', 
        required=False,
        default='MoS2',
        help='MoS2, WTe2, Graphene or BN'
    )

    parser.add_argument(
        '--intensity',
        required=False,
        default=4,
        help='Intensity 1, 2, 3 or 4'
    )
    
    args = parser.parse_args()

    train_model(args.reload_data_dir, args.starting_material)