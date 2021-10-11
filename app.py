"""
Transformer Seq2seq.
Put 7x7x1280 as the encoder input, and output HTML+CSS text as the decoder output

See html_SXN_parser/sxn_parser.py's comment to see more explaination related to parsing and more implementation strategy

Author: Samuel Koesnadi 2019

Attention weights naming:
decoder_layer4_block2 means 4th layer (from maximum num_layers) and second block (from the two blocks that decoder has)
"""
import os
import subprocess
import tarfile

import gradio as gr
import tensorflow as tf
from transformers import LEDConfig

from image2source.common_definitions import led_partial_configs, TARGET_SIZE, decoder_max_length
from image2source.sxn_parser import decode_sxn_folder
from image2source.utils import load_tokenizer_from_path, TFImageLEDForConditionalGeneration


pretrained_weights_dir = "pretrained_weights"

tokenizer_target_path = os.path.join(pretrained_weights_dir, "mockupai_tokenizer.json")
model_checkpoint_path = os.path.join(pretrained_weights_dir, "checkpoint")

# Download pretrained weights if does not exist
if not os.path.exists(pretrained_weights_dir):
    # Download the pretrained_weights
    bashCommand = (
        "curl https://api.github.com/repos/samuelmat19/image2source-tf2/releases/latest"
        " | grep \"browser_download_url\" | grep -Eo 'https://[^\"]*' | xargs wget")
    subprocess.run([bashCommand], shell=True, check=True, timeout=600)

    # Unzip it
    fname = "pretrained_weights.tar.xz"
    tar = tarfile.open(fname, "r:xz")
    tar.extractall()
    tar.close()

tokenizer = load_tokenizer_from_path(tokenizer_target_path)

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
    ckpt, model_checkpoint_path, max_to_keep=1)

epochs_trained = 0
steps_trained_in_current_epoch = 0
if led.ckpt_manager.latest_checkpoint:
    ckpt.restore(led.ckpt_manager.latest_checkpoint).expect_partial()
    print(led.ckpt_manager.latest_checkpoint, "is loaded")
else:
    print("Checkpoint is not loaded")


def process_image(input_img):
    img = tf.keras.applications.efficientnet.preprocess_input(input_img)
    img = tf.expand_dims(img, 0)

    output_sequences = led.generate(
        input_ids=img,
        max_length=decoder_max_length,
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

    raw_sxn_string = tokenizer.sequences_to_texts(output_sequences.numpy())[0]

    # filter the <s> </s> <sxn>
    filter_words = {"<s>", "</s>", "<sxn>"}
    sxn_string = ' '.join([word for word in raw_sxn_string.split(' ') if word not in filter_words])
    results = decode_sxn_folder(sxn_string)

    return results.get("index.html"), results


iface = gr.Interface(
    process_image,
    gr.inputs.Image(
        shape=(TARGET_SIZE, TARGET_SIZE),
        # image_mode="L", source="canvas"
        label="Website sketch"
    ),
    [
        gr.outputs.HTML(label="HTML (no CSS)"),
        gr.outputs.JSON(label="Files generated")
    ],
    # title="Mockup AI",
    # description="It will write HTML code based on the reference image inputted. Built with love by ML6",
    # server_port=8080
    enable_queue=True,
    live=True,

    interpretation="default",
    examples=[
        ["examples/example_1.png"],
        ["examples/example_2.png"],
        ["examples/example_3.png"],
        ["examples/example_4.png"],
    ]
)

iface.launch()
