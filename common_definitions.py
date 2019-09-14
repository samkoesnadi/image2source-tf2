import logging
import tensorflow as tf
LABEL_SMOOTHING_EPS = None

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

TOP_K = 20000  # this is for tokenizer

# maximum sequence length for dataset and transformer at evaluation time
MAX_SEQ_LEN_DATASET = 1053
MAX_SEQ_LEN = 1053

ACTIVATION = tf.nn.leaky_relu
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

### Set default parameters for all model
IMAGE_INPUT_SIZE = 224  # this to fit default criteria from MobileNetV2
BATCH_SIZE_AUTOENCODER = 64
BATCH_SIZE = 10
BUFFER_SIZE = 1000  # this is important for shuffling
EPOCHS = 100
BEAM_SEARCH_N = 3
DROPOUT_RATE = 0.1
REGULARIZER_RATE = 0.01
# LABEL_SMOOTHING_EPS = 0.1
MIN_EPOCH_TO_BREAK = 40
GAP_OF_DEAD_EPOCH = 50  # gap before it is going to kill the no more training network
DEFAULT_LEARNING_RATE = 1e-3
WARM_UP_STEPS = 4000  # for scheduler
# LEARNING_RATE_DECAY = (DEFAULT_LEARNING_RATE - 4e-4) / EPOCHS / 100

IS_TRAINING = True
IS_TEST_IMAGE = False
TRANSFER_LEARN_AUTOENCODER = True

USE_GPU = True

TARGET_FILENAME = "export.png"

ANNOTATIONS_PATH = "annotations.json"

# Normal parser
# TFRECORD_FILENAME = "datasets/pix2code.tfrecord"  # this has good result!
# TOKENIZER_FILENAME = "datasets/pix2code_tokenizer.json"  # this has good result!
# ADDITIONAL_FILENAME = "datasets/pix2code_additional.json"

# Harder parser
TFRECORD_FILENAME = "datasets/pix2code_harder.tfrecord"
TOKENIZER_FILENAME = "datasets/pix2code_tokenizer_harder.json"
ADDITIONAL_FILENAME = "datasets/pix2code_additional_harder.json"

MOBILENETV2_WEIGHT_PATH = "model_weights/pix2code_MobileNetV2.h5"  # autoencoder trained on pix2code datasets
TRANSFORMER_WEIGHT_PATH = "model_weights/pix2code_transformer.h5"  # transformer trained on pix2code datasets

AUTOENCODER_CHECKPOINT_PATH = "./checkpoints/train/autoencoder"

# Normal parser
# TRANSFORMER_CHECKPOINT_PATH = "./checkpoints/train/transformer_transfer_learning"  # this has good result!

# Harder parser
TRANSFORMER_CHECKPOINT_PATH = "./checkpoints/train/transformer_transfer_learning_harder_bottleneck"


### Set Hyperparameters for Transformer
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8



logging.basicConfig(level=LOGGING_LEVEL)
