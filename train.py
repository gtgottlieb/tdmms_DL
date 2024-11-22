"""
Module to train the AI

How to run from a terminal:
    1. activate your environment
    2. run: $ py train.py 
        with optional arguments:   
            --reload_data_dir <True or False> 
            --starting_material <MoS2, WTe2, Graphene or BN>
            --intensity <1, 2, 3 or 4>
            --last_layers <True or False>
"""

import os
import sys
import argparse
import datetime
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf

from imgaug import augmenters as iaa
from tdmms.tdmcoco import CocoConfig
from bep.utils import (
    check_dir_setup, 
    create_dir_setup,
    load_train_val_datasets,
    load_tdmms_weights
)
from bep.callbacks import MeanAveragePrecisionCallback
from notifications.discord import notify

ROOT_DIR = os.path.abspath("../")
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from mrcnn import model as modellib

tf.random.set_seed(42)

BATCH_SIZE = 2

#--------------------------------------------------------------#
#                         SETUP GPU                            #
#--------------------------------------------------------------#

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#--------------------------------------------------------------#
#                       SETUP LOGS DIR                         #
#--------------------------------------------------------------#

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs', 'training')

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)
    print(f"Folder '{DEFAULT_LOGS_DIR}' created.")

#--------------------------------------------------------------#
#                         TRAINING                            #
#--------------------------------------------------------------#

class TrainingConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = BATCH_SIZE

    def __init__(
        self,
        train_images: int,
        val_images: int,
        starting_material: str,
        intensity: int,
        last_layers: bool,
    ):
        super().__init__()
        batch_size = self.GPU_COUNT * self.IMAGES_PER_GPU
        total_image_count = train_images + val_images
        self.STEPS_PER_EPOCH = train_images / batch_size
        # Checkpoint name format:
        # <datetime now>_<fine-tuned on>_<fine-tuned from>_<images in train and validation set>_<intensity>_<batch size>_<epoch amount>
        
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.CHECKPOINT_NAME = f'{date}_nbse2_{starting_material.lower()}_{intensity}_{last_layers}_{total_image_count}_{batch_size}_'
        self.NAME = self.CHECKPOINT_NAME[:-1]

class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'inference'

def train_model(
    reload_data_dir: bool = False,
    starting_material: str = 'MoS2',
    intensity: int = 4,
    last_layers: bool = False
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
        - last_layers: True or False. Determines if the last layers are trained or not.
            Requires a matching amount of classes between transfer and new model.
    
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
        intensity,
        last_layers,
    )
    config.display()
    notify('Started training {}'.format(config.CHECKPOINT_NAME[:-1]))

    model = modellib.MaskRCNN(
        mode="training",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    MODEL_PATH = os.path.join(ROOT_DIR, 'weights', load_tdmms_weights(starting_material))

    print("Loading weights ", MODEL_PATH)

    if last_layers == 'True':
        # Amount of classes of transer model must be the same as the new model
        model.load_weights(MODEL_PATH, by_name=True)
    else:
        model.load_weights(MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    inference_config = InferenceConfig()
    model_inference = modellib.MaskRCNN(
        mode="inference",
        config=inference_config,
        model_dir=DEFAULT_LOGS_DIR
    )

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

    mean_average_precision_callback = MeanAveragePrecisionCallback(
        model,
        model_inference,
        dataset_val,
        calculate_map_at_every_X_epoch=5,
        verbose=1
    )


    if intensity >= 1:
        # Training - Stage 1
        notify('Training network heads {}'.format(config.CHECKPOINT_NAME[:-1]))
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads',
            augmentation=augmentation,
            custom_callbacks=[mean_average_precision_callback],
        )

    if intensity >= 2:    
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        notify('Fine tune Resnet stage 4 and up {}'.format(config.CHECKPOINT_NAME[:-1]))
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=60,
            layers='4+',
            augmentation=augmentation,
            custom_callbacks=[mean_average_precision_callback],
        )
    
    if intensity >= 3:
        # Training - Stage 3
        # Fine tune all layers
        notify('Fine tune all layers {}'.format(config.CHECKPOINT_NAME[:-1]))
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE /10,
            epochs=90,
            layers='all',
            augmentation=augmentation,
            custom_callbacks=[mean_average_precision_callback],
        )
    
    notify('Reduce LR and further tune all layers {}'.format(config.CHECKPOINT_NAME[:-1]))
    if intensity >= 4:
        notify('Reduce LR and further tune all layers {}'.format(config.CHECKPOINT_NAME[:-1]))
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE /100,
            epochs=120,
            layers='all',
            augmentation=augmentation,
            custom_callbacks=[mean_average_precision_callback],
        )

    notify('Done training')

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
        default=1,
        help='Intensity 1, 2, 3 or 4'
    )

    parser.add_argument(
        '--last_layers',
        required=False,
        default=False,
        help='True or False'
    )
    
    args = parser.parse_args()

    try:
        train_model(
            args.reload_data_dir,
            args.starting_material,
            int(args.intensity),
            args.last_layers
        )
    except Exception as e:
        logging.error("An exception occurred", exc_info=True)
        notify("An error occurred during training:\n{}".format(e))