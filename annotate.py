"""
Select and list all the datasets are wanted for the project. They would be stored in a dictionary of html_id and image_id. This dictionary is stored in json file.
Author: Samuel Koesnadi 2019
"""
import os

from image2source.dataset_helper import convert_and_write_all_datasets, store_tokenizer_to_path, \
    store_additional_info
from image2source.utils import dump_json_to_path, pairs_from_parent_path
from image2source.common_definitions import ORIGINAL_DATASET_PATH, ANNOTATIONS_PATH, \
    TFRECORD_FILENAME, TOKENIZER_FILENAME, ADDITIONAL_FILENAME


if __name__ == "__main__":
    ### MAKE PARENT DIRECTORY OF GENERATED FILES ###
    parent_dir = os.path.dirname(TFRECORD_FILENAME)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    # ### ANNOTATE PIX2CODE'S DATASETS ###
    # dump_json_to_path(ANNOTATIONS_PATH, pairs_from_parent_path(ORIGINAL_DATASET_PATH))

    ### CONVERT THE DATASETS TO TFRECORD ###
    print("convert_and_write_all_datasets")
    tokenizer, max_position = convert_and_write_all_datasets(
        ANNOTATIONS_PATH, TFRECORD_FILENAME)  # convert and write all datasets from the annotations path to TFRecord File

    # store Tokenizer object to path
    print("store_tokenizer_to_path")
    store_tokenizer_to_path(tokenizer, TOKENIZER_FILENAME)

    # store additional data to path
    print("store_additional_info")
    store_additional_info({"max_pos": max_position}, ADDITIONAL_FILENAME)
