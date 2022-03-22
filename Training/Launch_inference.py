import os
import sys
import random

from Object import ObjectConfig, ObjectDataset, ObjectInfer

# Root directory of the project
ROOT_PATH = os.path.abspath("./../../../")

# Import Mask RCNN
sys.path.append(ROOT_PATH)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
from visualize import *

##################################### CONFIGURATION #####################################
# Directory to save logs and trained model
MODEL_PATH = os.path.join(ROOT_PATH, "Logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")

# Path to the dataset
DATASET_PATH = os.path.join(ROOT_PATH, 'Real_dataset')

# Path to saved weights
OBJECT_MODEL_PATH = os.path.join(MODEL_PATH, 'mask_rcnn_object.h5')

# Path to save visualization figure
FIGURE_PATH = os.path.join(ROOT_PATH, 'Figures')

# Configuration file for training and inference
config = ObjectConfig()
config.display()
#########################################################################################

# Load a dataset
dataset_val = ObjectDataset()
dataset_val.load_object(DATASET_PATH, "val")
dataset_val.prepare()

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)


model = ObjectInfer(config=config, model_dir=MODEL_PATH)
model.load_weights(weights_dir=OBJECT_MODEL_PATH)
results = model.infer([original_image])

r = results[0]
display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], out_dir=FIGURE_PATH, name='test')