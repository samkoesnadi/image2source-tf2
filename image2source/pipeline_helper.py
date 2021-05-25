"""
Pipeline as the aggregator
"""
import logging
import math
from time import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tqdm import tqdm

from image2source.common_definitions import (
    IMAGE_FEATURE_DIMS,
    NUM_LAYERS_N,
    D_MODEL_N,
    DFF_N,
    NUM_HEADS_N, DROPOUT_RATE, WARM_UP_STEPS, FOCAL_LOSS, MAX_SEQ_LEN_DATASET, BEAM_SEARCH_N,
    MAX_SEQ_LEN, LOGGING_LEVEL,
)

from image2source.dataset_helper import (
    load_tokenizer_from_path,
    load_additional_info,

)
from image2source.html_SXN_parser.parser import decode_2_html

from image2source.transformers_helper import (
    Transformer, create_masks, create_look_ahead_mask
)
from image2source.utils import CustomSchedule, FocalLoss, SmartCheckpointSaver, save_fig_png


class Pipeline:
    """The main class that runs the generator"""

    def __init__(self, tokenizer_filename, additional_filename, checkpoint_path):
        # load tokenizer
        self.tokenizer = load_tokenizer_from_path(tokenizer_filename)

        # load additional info
        additional_info = load_additional_info(additional_filename)
        self.max_position = additional_info["max_pos"]

        self.target_vocab_size = len(self.tokenizer.index_word)  # the total length of index

        # instance of Transformer
        self.transformer = Transformer(NUM_LAYERS_N,
                                       D_MODEL_N,
                                       NUM_HEADS_N,
                                       DFF_N,
                                       IMAGE_FEATURE_DIMS,
                                       self.target_vocab_size,
                                       DROPOUT_RATE,
                                       self.max_position)

        # preprocessing base model
        self.image_feature_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False, weights=None, pooling=None)

        # define optimizer and loss
        self.learning_rate = CustomSchedule(DFF_N, WARM_UP_STEPS)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9, amsgrad=True, clipnorm=1.)

        if not FOCAL_LOSS:
            self.loss_object_sparse = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')
        else:
            self.loss_object_ = FocalLoss()

        # define train loss and accuracy
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # checkpoint
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                        optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=100)

        self.smart_ckpt_saver = SmartCheckpointSaver(self.ckpt_manager)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def load_weights(self, mobilenet_weight_path, transformer_weight_path):
        # load weights of the model
        self.image_feature_model.load_weights(mobilenet_weight_path)

        # preload to construct graph
        self.transformer(
            np.zeros((1, 49, 1280)),
            np.zeros((1, 1)),
            training=True, look_ahead_mask=None, decode_pos=None)
        self.transformer.load_weights(transformer_weight_path)

    def loss(self, real, pred, position, mask):
        """
        The loss is normalized iterated in each batch; divided by the masked sequence length
        in each batch

        :param mask:
        :param real:
        :param pred:
        :param position: if position > 0, then mask all loss except the last one
            (make sure that it is not padding)
        :return:
        """
        if self.max_position != 0:
            mask_pos = tf.map_fn(lambda pos: tf.ones(MAX_SEQ_LEN_DATASET - 1,
                                                     dtype=tf.dtypes.bool) if pos == 0 else tf.cast(
                tf.one_hot(MAX_SEQ_LEN_DATASET - 2, MAX_SEQ_LEN_DATASET - 1), dtype=tf.dtypes.bool)
                                 , position, dtype=tf.dtypes.bool)
            mask = tf.math.logical_and(mask, mask_pos)

        if not FOCAL_LOSS:
            loss_ = self.loss_object_sparse(real, pred)
        else:
            # FOCAL LOSS
            # convert real to one-hot encoding... this is to apply label smoothing
            real_one_hot = tf.one_hot(real, self.target_vocab_size)

            # # label smoothing
            # num_classes = tf.cast(tf.shape(real_one_hot)[1], pred.dtype)
            # smooth_positives = 1.0 - LABEL_SMOOTHING_EPS
            # smooth_negatives = LABEL_SMOOTHING_EPS / num_classes
            # real_one_hot = real_one_hot * smooth_positives + smooth_negatives

            loss_ = self.loss_object_(real_one_hot, pred)  # loss

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        # # normalize the loss_ for each batch
        # count_one_mask = tf.math.count_nonzero(mask, -1, keepdims=True, dtype=tf.float32)
        # loss_ /= count_one_mask

        return tf.reduce_sum(loss_)

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. TODO if possible: To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.
    @tf.function
    def train_step(self, img, sxn_token, decode_pos):
        tar_inp = sxn_token[:, :-1]
        tar_real = sxn_token[:, 1:]

        _mask = create_masks(tar_inp)

        mask = tf.math.logical_not(tf.math.equal(tar_real, 0))

        inp = self.image_feature_model(img)  # preprocess the image
        tf.print(tf.shape(inp))
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp,
                                              True,
                                              _mask,
                                              decode_pos)
            loss = self.loss(tar_real, predictions, decode_pos, mask)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions, mask)

    def evaluate(self, img, plot_layer=False):
        """
        TODO: implement window size in it
        :param plot_layer: Boolean to plot the intermediate layers
        :param img: (height, width, 3)
        :return:
        """
        start_token = self.tokenizer.word_index['<start>']
        end_token = self.tokenizer.word_index['<end>']

        # preprocessing
        img_expand_dims = tf.expand_dims(img, 0)
        encoder_input = self.image_feature_model(
            img_expand_dims)  # preprocessing_model needs to come in batch
        encoder_output = self.transformer.encoder(encoder_input, False,
                                                  None)  # (batch_size, inp_seq_len, d_model)

        # For beam search, tile encoder_output
        encoder_output = tf.tile(encoder_output, tf.constant([BEAM_SEARCH_N, 1, 1]))

        if plot_layer:
            # generate plot of encoder_input
            # store the figures
            for i, layer in enumerate([encoder_input]):
                save_fig_png(layer, "transformer/prediction_encoder_input_feature_layer_" + str(i))

        # as the target is english, the first word to the transformer should be the
        # english start token.
        beam_output = tf.expand_dims([start_token] * BEAM_SEARCH_N, -1)
        beam_prob = tf.expand_dims([1] * BEAM_SEARCH_N, -1)
        beam_result = None
        attention_weights = None

        for _ in tqdm(range(MAX_SEQ_LEN + self.max_position)):
            look_ahead_mask = create_look_ahead_mask(tf.shape(beam_output)[1])

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_output,
                                                              beam_output,
                                                              False,
                                                              look_ahead_mask,
                                                              None)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (BEAM_SEARCH_N, 1, vocab_size)

            predictions = tf.nn.softmax(predictions)  # softmax the output

            predictions = tf.reshape(predictions, [BEAM_SEARCH_N, self.target_vocab_size])

            # candidates and put it to beam_output
            candidates = predictions * tf.cast(beam_prob, tf.float32)
            candidates = tf.reshape(candidates, [-1])

            candidates_index = tf.range(self.target_vocab_size)
            candidates_index = tf.tile(tf.expand_dims(candidates_index, axis=0), [BEAM_SEARCH_N, 1])

            top_k_beams = tf.math.top_k(candidates, BEAM_SEARCH_N)
            top_k_beams_index = top_k_beams[1]
            i_beams = top_k_beams_index // self.target_vocab_size
            j_beams = top_k_beams_index - i_beams * self.target_vocab_size
            ij = tf.stack((i_beams, j_beams), axis=-1)

            a_beam = tf.gather_nd(beam_output, tf.expand_dims(ij[:, 0], axis=-1))
            b_beam = tf.gather_nd(candidates_index, tf.expand_dims(ij, axis=1))

            beam_output = tf.concat([a_beam, b_beam], axis=-1)

            # update beam probabilities
            beam_prob_pre = top_k_beams[0]
            beam_prob = tf.expand_dims(beam_prob_pre, axis=-1)

            predicted_beam_id = tf.cast(tf.argmax(beam_prob, axis=0)[0],
                                        tf.int32)  # greedily take maximum prediction
            beam_result = beam_output[predicted_beam_id]

            # return the result if the predicted_id is equal to the end token
            if beam_result[-1] == end_token:
                return beam_result[:-1], attention_weights

        # return the result if the predicted_id is equal to the end token
        if beam_result[-1] == end_token:
            return beam_result[:-1], attention_weights
        else:
            return beam_result, attention_weights

    def plot_attention_weights(self, attention, input, sxn_token, layer, filename, max_len=10):
        """

        :param max_len: maximum length for sequence of input and sxn_result. Keep this to small value
        :param attention:
        :param input: (49)
        :param result: sxn token (seq_len_of_sxn_token)
        :param layer:
        :return:
        """
        fig = plt.figure(figsize=(16, 8))

        attention = tf.squeeze(attention[layer], axis=0)

        # Truncate length to max_len
        attention = tf.slice(attention, [0, 0, 0], [-1, max_len, max_len])  # slice the tensor
        input = input[:max_len]
        sxn_token = sxn_token[:max_len]

        # temp var
        row = math.ceil(attention.shape[0] ** .5)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(row, row, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(input)))
            ax.set_yticks(range(len(sxn_token)))

            ax.set_ylim(len(sxn_token) - 1.5, -0.5)

            ax.set_xticklabels(
                list(map(str, input)),
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels(
                list(map(lambda i: self.tokenizer.index_word[i], sxn_token)),
                fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def calculate_accuracy(self, target, result, position):
        """

        :param target:
        :param result:
        :param position:
        :return: >=0 then accuracy score, -1 result size is smaler than target
        """
        len_target = target.size

        if len_target + position <= result.size:
            return accuracy_score(target, result[position:len_target + position])
        else:
            return -1

    def translate_from_dataset(self, test_data, plot=''):
        """

        :param test_data:
        :param plot:
        :return:
        """

        try:
            img = test_data[0]  # (height, width, 3)
            target = test_data[1].numpy()
            position = test_data[2].numpy()

            if LOGGING_LEVEL == logging.DEBUG:
                start_time = time()
            result, attention_weights = self.evaluate(img)
            if LOGGING_LEVEL == logging.DEBUG:
                end_time = time()

            result = result.numpy()  # convert to numpy
            result = result[1:]  # [1:] key is to remove the <start> token

            predicted_sxn = self.tokenizer.sequences_to_texts([result])[
                0]  # translate_from_dataset to predicted_sxn
            predicted_html = decode_2_html(
                predicted_sxn)  # translate_from_dataset to predicted html

            if LOGGING_LEVEL == logging.DEBUG:
                full_end_time = time()

                if plot:
                    self.plot_attention_weights(attention_weights, [i for i in range(49)], result,
                                                plot,
                                                "layers_figure/transformer/last_attention_weights.png")
                    print("Plot attention weight is generated.")

                accuracy = self.calculate_accuracy(target, result, position)  # print accuracy score
                if accuracy >= 0:
                    print("Accuracy score: {}".format(
                        accuracy))  # print accuracy score of both the sequence
                else:
                    print("Result size is smaler than target")

                print("Time spent inference of network: {}".format(end_time - start_time))
                print("Time spent translating: {}".format(full_end_time - start_time))

            return predicted_html

        except:
            return "<html></html>"  # this means bad translation

    def translate(self, img):
        result, attention_weights = self.evaluate(img)

        try:
            result = result.numpy()  # convert to numpy
            result = result[1:]  # [1:] key is to remove the <start> token

            predicted_sxn = self.tokenizer.sequences_to_texts([result])[
                0]  # translate_from_dataset to predicted_sxn
            predicted_html = decode_2_html(
                predicted_sxn)  # translate_from_dataset to predicted html

            return predicted_html
        except:
            return "<html></html>"  # this means bad translation
