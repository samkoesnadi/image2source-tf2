"""
Transformer Seq2seq.
Put 7x7x1280 as the encoder input, and output HTML+CSS text as the decoder output

See html_SXN_parser/parser.py's comment to see more explaination related to parsing and more implementation strategy

Author: Samuel Koesnadi 2019

Attention weights naming:
decoder_layer4_block2 means 4th layer (from maximum num_layers) and second block (from the two blocks that decoder has)
"""

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from image2source.common_definitions import TFRECORD_FILENAME, TRANSFORMER_CHECKPOINT_PATH, \
    TOKENIZER_FILENAME, IS_TRAINING, ADDITIONAL_FILENAME, EPOCHS, TRANSFORMER_WEIGHT_PATH, \
    IS_TEST_IMAGE, TARGET_FILENAME, MOBILENET_WEIGHT_PATH, D_MODEL_N
from image2source.dataset_helper import load_image
from image2source.pipeline_helper import Pipeline


if __name__ == "__main__":
    master = Pipeline(
        TOKENIZER_FILENAME, ADDITIONAL_FILENAME, TRANSFORMER_CHECKPOINT_PATH)  # master pipeline

    # load weights of the model
    master.load_weights(MOBILENET_WEIGHT_PATH, TRANSFORMER_WEIGHT_PATH)

    img = load_image(TARGET_FILENAME)

    predicted_html = master.translate(img)

    # write the html to file
    with open("generated/generated_" + TARGET_FILENAME + ".html", "w") as f:
        f.write(predicted_html)
