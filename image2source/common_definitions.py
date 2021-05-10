import logging
import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

LOGGING_LEVEL = logging.DEBUG
IS_TRAINING = True
IS_TEST_IMAGE = False
USE_GPU = False


TOP_K = 20000  # this is for tokenizer

# maximum sequence length for dataset and transformer at evaluation time
MAX_SEQ_LEN_DATASET = 1053
MAX_SEQ_LEN = 1053

ACTIVATION = tf.nn.relu6
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

### Set default parameters for all model
IMAGE_INPUT_SIZE = 224  # this to fit default criteria from EfficientNet
BATCH_SIZE = 1
BUFFER_SIZE = 200  # this is important for shuffling
EPOCHS = 300

BEAM_SEARCH_N = 3
N_TEST_DATASET = 3

# LABEL_SMOOTHING_EPS = 0.1
FOCAL_LOSS = False
SIGMOID = False

MIN_EPOCH_TO_BREAK = 40
GAP_OF_DEAD_EPOCH = 50  # gap before it is going to kill the no more training network
DEFAULT_LEARNING_RATE = 1e-3
WARM_UP_STEPS = 2000  # for scheduler
MIN_EPSILON = 1e-6
MAX_EPSILON = 1. - MIN_EPSILON
# LEARNING_RATE_DECAY = (DEFAULT_LEARNING_RATE - 4e-4) / EPOCHS / 100

# Focal loss parameter
ALPHA_BALANCED = 0.25
GAMMA_FOCAL = 1.0

### MODEL PARAMETERS ###
IMAGE_FEATURE_DIMS = 1280
NUM_LAYERS_N = 6
D_MODEL_N = 512
DFF_N = 2048
NUM_HEADS_N = 8
DROPOUT_RATE = 0.1
REGULARIZER_RATE = 0.01


### FILE PATHS ###

TARGET_FILENAME = "export.png"

ANNOTATIONS_PATH = "annotations.json"


# Harder parser
TFRECORD_FILENAME = "datasets/pix2code_better_extractor.tfrecord"
TOKENIZER_FILENAME = "datasets/pix2code_tokenizer_better_extractor.json"
ADDITIONAL_FILENAME = "datasets/pix2code_additional_better_extractor.json"

MOBILENETV2_WEIGHT_PATH = "model_weights/pix2code_MobileNetV2_better_extractor.h5"  # autoencoder trained on pix2code datasets
TRANSFORMER_WEIGHT_PATH = "model_weights/pix2code_transformer_better_extractor.h5"  # transformer trained on pix2code datasets

# Harder parser
TRANSFORMER_CHECKPOINT_PATH = "./checkpoints/train/transformer_better_extractor"

### ANNOTATE ORIGINAL DATASET ###
ORIGINAL_DATASET_PATH = "../datasets/pix2code"




######### MISCELLANEOUS #########
logging.basicConfig(level=LOGGING_LEVEL)

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
