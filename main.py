"""
Transformer Seq2seq.
Put 7x7x1280 as the encoder input, and output HTML+CSS text as the decoder output

See html_SXN_parser/sxn_parser.py's comment to see more explaination related to parsing and more implementation strategy

Author: Samuel Koesnadi 2019

Attention weights naming:
decoder_layer4_block2 means 4th layer (from maximum num_layers) and second block (from the two blocks that decoder has)
"""

import os
import time
from datetime import datetime

import tensorflow as tf
from matplotlib import pyplot as plt

from image2source.common_definitions import TFRECORD_FILENAME, TRANSFORMER_CHECKPOINT_PATH, \
    TOKENIZER_FILENAME, IS_TRAINING, ADDITIONAL_FILENAME, EPOCHS, TRANSFORMER_WEIGHT_PATH, \
    IS_TEST_IMAGE, TARGET_FILENAME
from image2source.dataset_helper import get_all_datasets, load_additional_info, \
    store_additional_info, load_image
from image2source.sxn_parser import decode_2_html
from image2source.pipeline_helper import Pipeline


if __name__ == "__main__":
    # initialize train dataset
    train_datasets, test_dataset = get_all_datasets(TFRECORD_FILENAME)
    key_epoch = "transformer_epoch_" + os.path.basename(
        TRANSFORMER_CHECKPOINT_PATH)  # the key name in additional info for prev epoch

    master = Pipeline(TOKENIZER_FILENAME, ADDITIONAL_FILENAME,
                      TRANSFORMER_CHECKPOINT_PATH)  # master pipeline

    if IS_TRAINING:
        # tensorboard support
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/transformer/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        ### Train loop
        start_epoch = 0
        additional_info = load_additional_info(ADDITIONAL_FILENAME)
        if master.ckpt_manager.latest_checkpoint:
            if key_epoch in additional_info:
                start_epoch = additional_info[key_epoch]
            else:
                start_epoch = additional_info["transformer_epoch"]

        total_batch_in_dataset = 0

        for epoch in range(start_epoch, EPOCHS):
            start = time.time()

            master.train_loss.reset_states()
            master.train_accuracy.reset_states()

            # inp -> image, tar -> html
            for (batch, (img, sxn_token, decode_pos)) in enumerate(train_datasets):
                master.train_step(img, sxn_token, decode_pos)

                if batch % 200 == 0:
                    print('Epoch {} Batch {} Loss {:e} Accuracy {:e}'.format(
                        epoch + 1, batch, master.train_loss.result(),
                        master.train_accuracy.result()))

                total_batch_in_dataset = batch

            total_batch_in_dataset += 1

            print('Epoch {}: Total batch {} Loss {:e} Accuracy {:e}'.format(epoch + 1,
                                                                            total_batch_in_dataset,
                                                                            master.train_loss.result(),
                                                                            master.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs'.format(time.time() - start))

            # store loss and acc to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', master.train_loss.result(),
                                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1
                tf.summary.scalar('accuracy', master.train_accuracy.result(), step=epoch)

            should_break = master.smart_ckpt_saver(epoch + 1,
                                                   master.train_accuracy.result())  # this will be better if we use validation
            if should_break == -1:
                start_epoch = epoch
                break
            elif should_break == 1:
                # store last epoch
                additional_info[key_epoch] = master.smart_ckpt_saver.max_acc_epoch
                store_additional_info(additional_info, ADDITIONAL_FILENAME)

            if (epoch + 1) % 50 == 0:
                # translate_from_dataset image to html for evaluation
                for i, test_data in enumerate(test_dataset):
                    print("Translating test index-" + str(i))
                    html = master.translate_from_dataset(test_data, "")

                    # store image for reference
                    plt.imshow(test_data[0])
                    plt.savefig('generated/transformer_input_img_{}.png'.format(i),
                                bbox_inches='tight')
                    plt.close()

                    # write the html to file
                    with open("generated/generated_" + str(i) + '_' + str(epoch + 1) + ".html",
                              "w") as f:
                        f.write(html)

                    # write the ground truth to file
                    with open("generated/ground_truth_" + str(i) + ".html", "w") as f:
                        true_sxn = master.tokenizer.sequences_to_texts([test_data[1].numpy()[1:]])[
                            0]  # translate_from_dataset to predicted_sxn
                        true_html = decode_2_html(
                            true_sxn)  # translate_from_dataset to predicted html
                        f.write(true_html)

            print()

        print(
            'Saving Transformer weights for epoch {}'.format(master.smart_ckpt_saver.max_acc_epoch))
        master.ckpt.restore(
            master.ckpt_manager.latest_checkpoint)  # load checkpoint that was just trained to model
        master.transformer.save_weights(TRANSFORMER_WEIGHT_PATH)  # save the preprocessing weights

    if IS_TEST_IMAGE:
        img = load_image(TARGET_FILENAME)

        predicted_html = master.translate(img)

        # write the html to file
        with open("generated/generated_" + TARGET_FILENAME + ".html", "w") as f:
            f.write(predicted_html)

    else:
        # evaluate
        print("Start evaluation...")

        # translate_from_dataset image to html
        for i, test_data in enumerate(test_dataset):
            print("Translating test index-" + str(i))
            html = master.translate_from_dataset(test_data, "")

            # store image for reference
            plt.imshow(test_data[0])
            plt.savefig('generated/transformer_input_img_{}.png'.format(i), bbox_inches='tight')
            plt.close()

            # write the html to file
            with open("generated/generated_" + str(i) + ".html", "w") as f:
                f.write(html)

            # write the ground truth to file
            with open("generated/ground_truth_" + str(i) + ".html", "w") as f:
                true_sxn = master.tokenizer.sequences_to_texts([test_data[1].numpy()[1:]])[
                    0]  # translate_from_dataset to predicted_sxn
                true_html = decode_2_html(true_sxn)  # translate_from_dataset to predicted html
                f.write(true_html)
