import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import tensorflow as tf
from transformers import TFLEDForConditionalGeneration, LEDConfig

from image2source.common_definitions import led_partial_configs, TARGET_SIZE, d_model, encoder_max_length, \
    decoder_max_length
from image2source.utils import load_tokenizer_from_path, TFImageLEDForConditionalGeneration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizerTargetPath", help="the target directory of the tokenizer",
        default="datasets/mockupai_tokenizer.json", type=str)
    parser.add_argument(
        "--modelCheckpointPath", help="the target directory of the checkpoint",
        default="outputs/led_2021-10-11-07:36:59.721503/checkpoint", type=str)
    parser.add_argument(
        "--fileUrl", help="the input image file",
        default="./datasets/parsed/datasets-raw-pix2code-7AFECB33-ADE7-476B-8027-C7F0C35427A5-html.png",
        type=str)
    args = parser.parse_args()

    tokenizer = load_tokenizer_from_path(args.tokenizerTargetPath)

    bos_token = (tokenizer.word_index["<s>"])
    sxn_token = (tokenizer.word_index["<sxn>"])
    eos_token = (tokenizer.word_index["</s>"])
    pad_token = (tokenizer.word_index["<pad>"])

    led = TFImageLEDForConditionalGeneration(
        LEDConfig(
            **led_partial_configs,
            use_cache=True,
            gradient_checkpointing=False,
            decoder_start_token_id=sxn_token,
            pad_token_id=pad_token,
            bos_token_id=bos_token,
            eos_token_id=eos_token,
        )
    )

    ckpt = tf.train.Checkpoint(model=led)
    led.ckpt_manager = tf.train.CheckpointManager(
        ckpt, args.modelCheckpointPath, max_to_keep=1)

    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if led.ckpt_manager.latest_checkpoint:
        ckpt.restore(led.ckpt_manager.latest_checkpoint).expect_partial()
        print(led.ckpt_manager.latest_checkpoint, "is loaded")
    else:
        print("Checkpoint is not loaded")

    # # prepare input
    # file_url = (
    #     "/media/radoxpi/Garage/project/AI_web_designer/datasets/pix2code/FEF248A4-868E-4A6C-94D6-9B38A67974F0.html",
    #     "/media/radoxpi/Garage/project/AI_web_designer/datasets/pix2code/FEF248A4-868E-4A6C-94D6-9B38A67974F0.png",
    # )

    # 2. read the image
    img = tf.io.read_file(args.fileUrl)
    img = tf.image.decode_png(img, channels=3)

    img = tf.image.resize(img, (TARGET_SIZE, TARGET_SIZE))

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    img = tf.expand_dims(img, 0)

    # encoded_img = encoded_img.numpy()
    # encoded_img = np.append(
    #     encoded_img,
    #     [np.zeros(encoded_img.shape[-1])] * (encoder_max_length - len(encoded_img) + 1), axis=0)
    # encoded_img = encoded_img[:-1, :].astype(np.float32)
    #
    # attention_mask = (np.sum(encoded_img, axis=-1) != 0).astype(np.float32)
    #
    # # the global attention mask is set to 0 because the attention window is 512
    # global_attention_mask = np.zeros(encoded_img.shape[:1], dtype=np.float32)

    # TODO
    output_sequences = led.generate(
        input_ids=img,
        max_length=decoder_max_length,
        top_p=0.9,
        do_sample=False,
        num_return_sequences=1,
        early_stopping=True,
        repetition_penalty=1.,
        bos_token_id=bos_token,
        pad_token_id=pad_token,
        eos_token_id=eos_token,
        decoder_start_token_id=sxn_token,

        # # model related
        # attention_mask=attention_mask,
        # global_attention_mask=global_attention_mask
    )

    print(tokenizer.sequences_to_texts(output_sequences.numpy()))
