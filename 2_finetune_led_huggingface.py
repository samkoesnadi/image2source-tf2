import argparse
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf
from transformers import TFTrainer, TFTrainingArguments, TFLEDForConditionalGeneration, LEDConfig, IntervalStrategy
import transformers

from datasets import load_metric

from image2source.common_definitions import led_partial_configs, TARGET_SIZE, d_model, encoder_max_length, \
    decoder_max_length
from image2source.utils import load_tokenizer_from_path, TFImageLEDForConditionalGeneration

transformers.logging.set_verbosity_debug()


def load_data(full_file_path):
    full_file_path = full_file_path.decode('UTF-8')
    png_full_path = full_file_path + '.png'
    sxn_full_path = full_file_path + '.sxn'

    # read the image
    img = tf.io.read_file(png_full_path)
    img = tf.image.decode_png(img, channels=3)

    # min_size = tf.reduce_min(img.shape[:2])
    # ratio = TARGET_SIZE / min_size
    #
    # img = tf.image.resize(
    #     img, (int(img.shape[0] * ratio), int(img.shape[1] * ratio)),
    #     preserve_aspect_ratio=True)
    # img = img[:TARGET_SIZE, :TARGET_SIZE, :]

    # Make it target_size, target_size
    img = tf.image.resize(img, (TARGET_SIZE, TARGET_SIZE))

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    # encoded_img = image_encoder(img, training=False)
    #
    # encoded_img = tf.reshape(
    #     encoded_img, (-1, encoded_img.shape[-1]))
    # encoded_img = encoded_img[:, :d_model]
    # encoded_img = encoded_img[:encoder_max_length]

    # read the sxn
    with open(sxn_full_path, 'r') as f:
        sxn = f.read()
    sxn_token = tokenizer.texts_to_sequences(['<s> ' + sxn + ' </s>'])[0]
    sxn_token = sxn_token[:decoder_max_length]

    # encoded_img = np.append(
    #     encoded_img,
    #     [np.zeros(encoded_img.shape[-1])] * (max_len_encoded_imgs - len(encoded_img) + 1), axis=0)
    # encoded_img = encoded_img[:-1, :].astype(np.float32)

    # this is flattened
    sxn_token = np.append(sxn_token, [0] * (decoder_max_length - len(sxn_token))).astype(np.int32)

    # attention_mask = (np.sum(encoded_img, axis=-1) != 0).astype(np.float32)
    #
    # # the global attention mask is set to 0 because the attention window is 512
    # global_attention_mask = np.zeros(encoded_img.shape[:1], dtype=np.float32)

    return img, sxn_token


def process_data_to_model_inputs(batch_encoded_image, batch_sxn_token):
    model_inputs = {
        "input_ids": batch_encoded_image,
        # "attention_mask": attention_mask,
        # "global_attention_mask": global_attention_mask
    }
    return model_inputs, batch_sxn_token


def get_tf_dataset(full_file_paths):
    train_dataset = tf.data.Dataset.from_tensor_slices((full_file_paths,))
    train_dataset = train_dataset.map(
        lambda x: tf.numpy_function(load_data, [x], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    train_dataset = train_dataset.cache()
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
        "--num_evals", help="the training eval size",
        default=4, type=int)
    parser.add_argument(
        "--num_epochs", help="the training epoch number",
        default=100, type=int)

    args = parser.parse_args()

    # setup models - BEWARE: this is used by the function outside of main
    tokenizer = load_tokenizer_from_path(args.tokenizerTargetPath)

    bos_token = (tokenizer.word_index["<s>"])
    eos_token = (tokenizer.word_index["</s>"])
    pad_token = (tokenizer.word_index["<pad>"])
    sxn_token = (tokenizer.word_index["<sxn>"])

    # read the filenames
    with open(os.path.join(args.targetDir, "website_stats.json"), 'r') as f:
        website_stats = json.load(f)
    full_file_paths = [
        os.path.join(args.targetDir, filename)
        for filename
        in [k for k, v in website_stats.items()]]
        # in [k for k, v in website_stats.items() if v <= decoder_max_length - 3]]

    raw_dataset = get_tf_dataset(full_file_paths)
    eval_dataset = raw_dataset.take(args.num_evals)
    train_dataset = raw_dataset.skip(args.num_evals)

    # 2. Train model
    led_config = LEDConfig(
        **led_partial_configs,
        use_cache=False,
        gradient_checkpointing=True,
        decoder_start_token_id=sxn_token,
        pad_token_id=pad_token,
        bos_token_id=bos_token,
        eos_token_id=eos_token,

        # evaluation metric related
        num_beams=3,
        max_length=decoder_max_length,
        min_length=50,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    # 3. Set evaluation metric
    rouge = load_metric("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = np.argmax(pred.predictions, axis=-1)

        label_str = tokenizer.sequences_to_texts(labels_ids)
        pred_str = tokenizer.sequences_to_texts(pred_ids)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    # process per bucket
    folder_name = str(datetime.now()).replace(' ', '-')
    run_name = f"led_{folder_name}"

    training_args = TFTrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        logging_steps=20,
        fp16=False,
        save_strategy=IntervalStrategy.EPOCH,  # this does not work
        gradient_accumulation_steps=16,
        num_train_epochs=args.num_epochs,
        logging_dir=f'./logs/{run_name}',
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=100,
        xla=True,
        # poly_power=0.5,
        learning_rate=1e-4,
        warmup_ratio=0.1
    )

    with training_args.strategy.scope():
        led = TFImageLEDForConditionalGeneration(led_config)

    trainer = TFTrainer(
        model=led,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    led.save_pretrained(args.saveModelDir)
