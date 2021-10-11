"""
Transformer Seq2seq.
Put 7x7x1280 as the encoder input, and output HTML+CSS text as the decoder output

See html_SXN_parser/sxn_parser.py's comment to see more explaination related to parsing and more implementation strategy

Author: Samuel Koesnadi 2019

Attention weights naming:
decoder_layer4_block2 means 4th layer (from maximum num_layers) and second block (from the two blocks that decoder has)
"""
import argparse
import logging
import os

import tensorflow as tf

from image2source.common_definitions import TRANSFORMER_CHECKPOINT_PATH, \
    TOKENIZER_FILENAME, ADDITIONAL_FILENAME, TRANSFORMER_WEIGHT_PATH, \
    TARGET_FILENAME, MOBILENET_WEIGHT_PATH
from image2source.dataset_helper import load_image_skimage
from image2source.pipeline_helper import Pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        default=TARGET_FILENAME,
        help="path of the input image",
        type=str
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    master = Pipeline(
        TOKENIZER_FILENAME, ADDITIONAL_FILENAME, TRANSFORMER_CHECKPOINT_PATH)  # master pipeline

    # load weights of the model
    master.load_weights(MOBILENET_WEIGHT_PATH, TRANSFORMER_WEIGHT_PATH)
    logging.debug("Weights are loaded!")

    img = load_image_skimage(args.filename)
    logging.debug("Target image is loaded!")

    logging.debug("Start converting!")
    predicted_html = master.translate(img)

    # write the html to file
    ### MAKE PARENT DIRECTORY OF GENERATED FILES ###
    if not os.path.exists("generated"):
        os.mkdir("generated")

    with tf.io.gfile.GFile("generated/generated_from_predict.html", "w") as f:
        f.write(predicted_html)
    logging.debug("HTML/CSS is generated!")
