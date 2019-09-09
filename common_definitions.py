import logging
logging.basicConfig(level=logging.DEBUG)

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

TOP_K = 20000  # this is for tokenizer
MAX_SEQ_LEN = 1050

ACTIVATION = tf.nn.leaky_relu
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

### Set default parameters for all model
IMAGE_INPUT_SIZE = 224  # this to fit default criteria from MobileNetV2
BATCH_SIZE_AUTOENCODER = 64
BATCH_SIZE = 10
BUFFER_SIZE = 3000  # this is important for shuffling
EPOCHS = 1000
DROPOUT_RATE = 0.1
# LABEL_SMOOTHING_EPS = 0.1
MIN_EPS_TO_BREAK = 10
DEFAULT_LEARNING_RATE = 1e-3
WARM_UP_STEPS = 2000  # for scheduler
# LEARNING_RATE_DECAY = (DEFAULT_LEARNING_RATE - 4e-4) / EPOCHS / 100

IS_TRAINING = True
TRANSFER_LEARN_AUTOENCODER = True

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
TRANSFORMER_CHECKPOINT_PATH = "./checkpoints/train/transformer_transfer_learning_harder"


### Set Hyperparameters for Transformer
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
