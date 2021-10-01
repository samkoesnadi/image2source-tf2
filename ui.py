"""
Transformer Seq2seq.
Put 7x7x1280 as the encoder input, and output HTML+CSS text as the decoder output

See html_SXN_parser/parser.py's comment to see more explaination related to parsing and more implementation strategy

Author: Samuel Koesnadi 2019

Attention weights naming:
decoder_layer4_block2 means 4th layer (from maximum num_layers) and second block (from the two blocks that decoder has)
"""
import argparse
import logging
import os

import gradio as gr
import numpy as np
import tensorflow as tf

from image2source.common_definitions import TRANSFORMER_CHECKPOINT_PATH, \
    TOKENIZER_FILENAME, ADDITIONAL_FILENAME, TRANSFORMER_WEIGHT_PATH, \
    TARGET_FILENAME, MOBILENET_WEIGHT_PATH, IMAGE_INPUT_SIZE
from image2source.dataset_helper import load_image_skimage
from image2source.pipeline_helper import Pipeline


if __name__ == "__main__":
    master = Pipeline(
        TOKENIZER_FILENAME, ADDITIONAL_FILENAME, TRANSFORMER_CHECKPOINT_PATH)  # master pipeline

    # load weights of the model
    master.load_weights(MOBILENET_WEIGHT_PATH, TRANSFORMER_WEIGHT_PATH)
    logging.debug("Weights are loaded!")

    def process_image(input_img):
        img = tf.image.resize(input_img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        predicted_html = master.translate(img)

        return predicted_html

    iface = gr.Interface(
        process_image, gr.inputs.Image(shape=(224, 224)), "text",
        title="Mockup AI",
        description="It will write HTML code based on the reference image inputted. Built with love by ML6",
        server_port=8080
    )

    iface.launch()
