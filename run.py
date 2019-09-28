"""
    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla
    
    # Train a new model starting from pre-trained COCO weights
    python3 run.py train --dataset=/path/to/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 run.py train --dataset=/path/to/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 run.py train --dataset=/path/to/dataset --weights=imagenet
    # Apply color mask to an image
    python3 run.py mask --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color mask to video using the last weights you trained
    python3 run.py mask --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "./mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the Hair dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Hair"

    # Running on CPU
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Hair

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        self.add_class("Hair", 1, "Hair")
        dataset_dir = os.path.join(dataset_dir, subset)
        for filename in os.listdir(os.path.join(dataset_dir, 'photos')):
            if not filename.endswith('jpg'): #Only jpg photos from dataset
                continue
            input_path = os.path.join(dataset_dir, 'photos',filename)
            img = cv2.imread(input_path)
            height, width = img.shape[:2]

            self.add_image(
                "Hair",  # for a single class just add the name here
                image_id=filename,  # use file name as a unique image id
                path= input_path,
                width=width, height=height)

    def load_mask(self,image_id):
        """Generate instance masks for an image from database.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "Hair":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info['id'].startswith('v'): # If validation mask or training
            dataset_dir = os.path.join(args.dataset, 'val')
        else:
            dataset_dir = os.path.join(args.dataset, 'train')
        for maskf in os.listdir(os.path.join(dataset_dir, 'masks')):
            mname,png = os.path.splitext(maskf)
            iname,jpg = os.path.splitext(info['id'])
            if mname == iname:
                mask=cv2.imread(os.path.join(dataset_dir, 'masks',maskf)) # Reading mask data from dataset

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Hair":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    
        # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=140,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=220,
                    layers='all')


def apply_mask(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    blank = np.zeros(image.shape, dtype=np.uint8)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        crop = np.where(mask, image, blank).astype(np.uint8)
    else:
        crop = blank.astype(np.uint8)
    return crop


def detect_and_mask(model, image_path=None, video_path=None):
    assert image_path or video_path

    if image_path:
        import cv2
        # Run model detection and generate the mask
        print("Running on {}".format(args.image))
        # Read image
        image = cv2.imread(args.image)
        # Detect objects
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        r = model.detect([image], verbose=1)[0]
        # Mask
        crop = apply_mask(image, r['masks'])
        # Save output
        crop = cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
        file_name = "crop_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        cv2.imwrite(file_name,crop)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "crop_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                crop = apply_mask(image, r['masks'])
                # RGB -> BGR to save image to video
                crop = crop[..., ::-1]
                # Add image to video writer
                vwriter.write(crop)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'mask'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color mask effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color mask effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "mask":
        assert args.image or args.video,\
               "Provide --image or --video to apply color mask"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "mask":
        detect_and_mask(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'mask'".format(args.command))
