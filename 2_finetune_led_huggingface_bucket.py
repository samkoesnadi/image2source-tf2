import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import TFTrainer, TFTrainingArguments, TFLEDForConditionalGeneration, LEDConfig
import transformers
transformers.logging.set_verbosity_debug()

from image2source.utils import load_tokenizer_from_path

d_model = 1024  # this has to match with the image encoder

TARGET_SIZE = 224
encoder_max_length = 128
decoder_max_length = 2048
tokenizer_max_vocab_size = 2750


def load_data(full_file_paths):
    encoded_imgs = []
    sxn_tokens = []

    for full_file_path in full_file_paths:
        full_file_path = full_file_path.decode('UTF-8')
        png_full_path = full_file_path + '.png'
        sxn_full_path = full_file_path + '.sxn'

        # read the image
        img = tf.io.read_file(png_full_path)
        img = tf.image.decode_png(img, channels=3)

        min_size = tf.reduce_min(img.shape[:2])
        ratio = TARGET_SIZE / min_size

        img = tf.image.resize(
            img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
        img = tf.keras.applications.efficientnet.preprocess_input(img)

        encoded_img = image_encoder(img, training=False)

        encoded_img = tf.reshape(
            encoded_img, (-1, encoded_img.shape[-1]))

        encoded_img = encoded_img[:, :d_model]

        encoded_imgs.append(encoded_img[:encoder_max_length])

        # read the sxn
        with open(sxn_full_path, 'r') as f:
            sxn = f.read()
        sxn_token = tokenizer.texts_to_sequences([sxn])[0]
        sxn_tokens.append(sxn_token[:decoder_max_length])


    max_len_encoded_imgs = np.max(list(map(len, encoded_imgs)))
    max_len_sxn_tokens = np.max(list(map(len, sxn_tokens)))

    # max_len_encoded_imgs = encoder_max_length
    # max_len_sxn_tokens = decoder_max_length

    encoded_imgs = np.array(
        [np.append(
            encoded_img,
            [np.zeros(encoded_img.shape[-1])] * (max_len_encoded_imgs - len(encoded_img) + 1), axis=0)
         for encoded_img in encoded_imgs], dtype=np.float32)
    encoded_imgs = encoded_imgs[:, :-1, :]

    # this is flattened
    sxn_tokens = np.array(
        [np.append(sxn_token, [0] * (max_len_sxn_tokens - len(sxn_token)))
         for sxn_token in sxn_tokens], dtype=np.int32)

    attention_mask = (np.sum(encoded_imgs != 0, axis=-1) != 0).astype(np.float32)
    attention_mask = np.ones_like(attention_mask)

    # the global attention mask is set to 0 because the attention window is 512
    global_attention_mask = np.zeros(encoded_imgs.shape[:2], dtype=np.float32)

    return encoded_imgs, sxn_tokens, attention_mask, global_attention_mask


def process_data_to_model_inputs(batch_encoded_image, batch_sxn_token, attention_mask, global_attention_mask):
    model_inputs = {
        "inputs_embeds": tf.stop_gradient(batch_encoded_image),
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask
    }
    return model_inputs, batch_sxn_token


def get_tf_dataset(full_file_paths, batch_size):
    full_file_paths = full_file_paths[: len(full_file_paths) // batch_size * batch_size]
    batched_full_file_paths = np.array_split(np.array(full_file_paths), len(full_file_paths) // batch_size)

    # IMPORTANT: shuffle the data per bucket
    [np.random.shuffle(x) for x in batched_full_file_paths]

    train_dataset = tf.data.Dataset.from_tensor_slices((batched_full_file_paths,))
    train_dataset = train_dataset.map(
        lambda x: tf.numpy_function(load_data, [x], [tf.float32, tf.int32, tf.float32, tf.float32]),
        # num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.map(
        process_data_to_model_inputs)
    return train_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sourceDir", help="the source directory of the parsed data", default="datasets/raw", type=str)
    parser.add_argument(
        "--targetDir", help="the target directory of the parsed data", default="datasets/parsed", type=str)
    parser.add_argument(
        "--saveModelDir", help="the target directory of the model",
        default="outputs/model", type=str)
    parser.add_argument(
        "--tokenizerTargetPath", help="the target directory of the tokenizer",
        default="datasets/mockupai_tokenizer.json", type=str)
    parser.add_argument(
        "--batch_size", help="the training batch size",
        default=2, type=int)
    parser.add_argument(
        "--n_bucket", help="the training n bucket",
        default=1967, type=int)
    parser.add_argument(
        "--inside_num_epochs", help="the training epoch number",
        default=50, type=int)
    parser.add_argument(
        "--outside_num_epochs", help="the training epoch number",
        default=10, type=int)

    args = parser.parse_args()

    batch_size = args.batch_size

    # setup models - BEWARE: this is used by the function outside of main
    image_encoder = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights='imagenet',
        pooling=None
    )
    tokenizer = load_tokenizer_from_path(args.tokenizerTargetPath)

    bos_token = (tokenizer.word_index["<s>"])
    eos_token = (tokenizer.word_index["</s>"])
    pad_token = (tokenizer.word_index["<pad>"])

    # read the filenames
    with open(os.path.join(args.targetDir, "website_stats.json"), 'r') as f:
        website_stats = json.load(f)
    full_file_paths = [
        os.path.join(args.targetDir, filename)
        for filename
        in [k for k, _ in sorted(website_stats.items(), key=lambda item: item[1])]]

    bucket_size = len(full_file_paths) // args.n_bucket

    train_dataset_bucket = get_tf_dataset(full_file_paths, bucket_size)

    # 2. Train model
    led_config = LEDConfig(
        vocab_size=tokenizer_max_vocab_size,
        encoder_layers=6,
        decoder_layers=6,
        decoder_ffn_dim=3072,
        encoder_ffn_dim=3072,
        max_encoder_position_embeddings=1024,
        max_decoder_position_embeddings=16384,
        num_hidden_layers=6,
        d_model=d_model,
        attention_window=[
            128,
            128,
            128,
            128,
            128,
            128
        ],
        use_cache=False,
        gradient_checkpointing=True,
        decoder_start_token_id=bos_token,
        pad_token_id=pad_token,
        bos_token_id=bos_token,
        eos_token_id=eos_token,
    )

    # process per bucket
    previous_checkpoint_dir = ""
    for epoch in tqdm(range(args.outside_num_epochs), desc="Epoch"):
        for train_dataset in tqdm(train_dataset_bucket, desc="Bucket"):
            len_sequence = train_dataset[1].shape[1]

            train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)

            for i in train_dataset.take(1):
                print(i)
                exit()

            folder_name = str(datetime.now()).replace(' ', '-')
            run_name = f"led_epoch_{epoch}_seqlen_{len_sequence}_{folder_name}"

            training_args = TFTrainingArguments(
                per_device_train_batch_size=batch_size,
                overwrite_output_dir=True,
                output_dir=f"outputs/{run_name}",
                run_name=run_name,
                logging_steps=1,
                fp16=False,
                save_strategy="epoch",
                save_steps=49,
                gradient_accumulation_steps=1,
                num_train_epochs=args.inside_num_epochs,
                logging_dir=f'./logs/{run_name}',
                do_train=True,
                do_eval=False,
                evaluation_strategy="no",
                eval_steps=1
            )

            # with training_args.strategy.scope():
            led = TFLEDForConditionalGeneration(led_config)

            temp_ckpt = tf.train.Checkpoint(model=led)
            led.ckpt_manager = tf.train.CheckpointManager(
                temp_ckpt, previous_checkpoint_dir, max_to_keep=10)

            if led.ckpt_manager.latest_checkpoint:
                temp_ckpt.restore(led.ckpt_manager.latest_checkpoint).expect_partial()

            temp_ckpt.ckpt_manager = None

            trainer = TFTrainer(
                model=led,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()

            led.save_pretrained(
                os.path.join(args.saveModelDir, str(epoch)))

            previous_checkpoint_dir = training_args.output_dir
