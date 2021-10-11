import argparse
import json
import os

import tensorflow as tf

from image2source.utils import store_tokenizer_to_path


def add_token(tokenizer, tag):
    new_id = len(tokenizer.index_word) + 1
    tokenizer.word_index[tag] = new_id
    tokenizer.index_word[new_id] = tag

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sourceDir", help="the source directory of the parsed data", default="datasets/raw", type=str)
    parser.add_argument(
        "--targetDir", help="the target directory of the parsed data", default="datasets/parsed", type=str)
    parser.add_argument(
        "--tokenizerTargetPath", help="the target directory of the tokenizer",
        default="datasets/mockupai_tokenizer.json", type=str)
    args = parser.parse_args()

    # read the filenames
    with open(os.path.join(args.targetDir, "website_stats.json"), 'r') as f:
        website_stats = json.load(f)
    full_file_paths = [
        os.path.join(args.targetDir, filename + '.sxn') for filename in website_stats.keys()]

    sxns = []
    for full_file_path in full_file_paths:
        with open(full_file_path, 'r') as f:
            sxns.append(f.read())

    # tokenize the sxns
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        100_000, filters='', split=' ', oov_token="oov")
    tokenizer.fit_on_texts(sxns)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    tokenizer = add_token(tokenizer, "<s>")
    tokenizer = add_token(tokenizer, "</s>")
    tokenizer = add_token(tokenizer, "<img>")
    tokenizer = add_token(tokenizer, "</img>")
    tokenizer = add_token(tokenizer, "<sep>")
    tokenizer = add_token(tokenizer, "<sxn>")
    tokenizer = add_token(tokenizer, "</sxn>")

    print("Vocab size:", len(tokenizer.word_index))

    # store Tokenizer object to path
    print("store_tokenizer_to_path")
    store_tokenizer_to_path(tokenizer, args.tokenizerTargetPath)
