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

TOP_K = 2000  # this is for tokenizer
MAX_SEQ_LEN = 1000

ACTIVATION = tf.nn.leaky_relu
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

### Set default parameters for all model
IMAGE_INPUT_SIZE = 224  # this to fit default criteria from MobileNetV2
BATCH_SIZE = 8
BUFFER_SIZE = 1000  # this is important for shuffling
EPOCHS = 5
DROPOUT_RATE = 0.1
# LEARNING_RATE = 1e-3  # Use Custom scheduler instead
# LEARNING_RATE_DECAY = (LEARNING_RATE - 4e-4) / EPOCHS / 100

IS_TRAINING = True

TFRECORD_FILENAME = "datasets/pix2code.tfrecord"
TOKENIZER_FILENAME = "datasets/pix2code_tokenizer.json"
ADDITIONAL_FILENAME = "datasets/pix2code_additional.json"
ANNOTATIONS_PATH = "annotations.json"
MOBILENETV2_WEIGHT_PATH = "model_weights/pix2code_MobileNetV2.h5"
DATASET_CACHE = "/tmp/img2source_dataset_cache"

### Set Hyperparameters for Transformer
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
