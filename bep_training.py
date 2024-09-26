"""Module to train the AI"""

import os
import sys
import argparse
from imgaug import augmenters as iaa

from tdmcoco import CocoConfig, evaluate_coco

from bep_data import bepDataset
from bep_utils import check_dir_setup

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'weights', 'graphene_mask_rcnn_tdm_0120.h5') # Graphene COCO+2D
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

def train_model():
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'train' or 'evaluate' on MS COCO"
    )
    parser.add_argument(
        '--model',
        required=False,
        default='coco',
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'"
    )
    parser.add_argument(
        '--logs',
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help='Logs and checkpoints directory (default=logs/)'
    )
    parser.add_argument(
        '--excludelastlayers',
        required=False,
        default=False,
        metavar="<True|False>",
        help="Exclude last layers during loading weights",
        type=bool
    )
    
    args = parser.parse_args()
    
    print("Command: ", args.command)
    print("Model: ", args.model)
    
    if args.command == "train":
        class TrainingConfig(CocoConfig):
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
        config = TrainingConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    # elif args.model.lower() == "last":
    #     # Find last trained weights
    #     model_path = model.find_last()
    # elif args.model.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     model_path = model.get_imagenet_weights()
    # else:
    #     model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    
    if (args.model.lower() == "coco") or args.excludelastlayers:
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)
        
    if args.command == "train":
        check_dir_setup(ROOT_DIR, 0.7)
        
        dataset_train = bepDataset()
        dataset_train.load_dir(os.path.join(ROOT_DIR, 'data'), 'train')
        dataset_train.prepare()

        dataset_val = bepDataset()
        dataset_val.load_dir(os.path.join(ROOT_DIR, 'data'), 'val')
        dataset_val.prepare()

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
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads',
                    augmentation=augmentation)
        
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=60,
                    layers='4+',
                    augmentation=augmentation)
        
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE /10,
                    epochs=90,
                    layers='all',
                    augmentation=augmentation)
        
        print("Reduce LR and further tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE /100,
                    epochs=120,
                    layers='all',
                    augmentation=augmentation)
             
    elif args.command == "evaluate":
        dataset_val = bepDataset()
        coco = dataset_val.load_dir(os.path.join(ROOT_DIR, 'data'), 'val', return_coco=True)
        dataset_val.prepare()

        print("Running COCO evaluation on {} images.".format(args.limit))
        model.keras_model.save("mrcnn_eval.h5")
        json_string = model.keras_model.to_json()
        with open('mrcnn_eval.json', 'w') as f:
            f.write(json_string)
            
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

    
    
if __name__ == '__main__':
    train_model()