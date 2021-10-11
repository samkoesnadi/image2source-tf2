import json
from typing import Optional, Union

import tensorflow as tf
from transformers.generation_tf_utils import TFGreedySearchOutput, TFBeamSearchOutput, TFSampleOutput, \
    TFBeamSampleOutput, shape_list, logger
from transformers.modeling_tf_utils import input_processing
from transformers.models.led.modeling_tf_led import TFLEDSeq2SeqLMOutput, shift_tokens_right, \
    TFLEDEncoderBaseModelOutput, TFLEDForConditionalGeneration


def _tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer


def load_tokenizer_from_path(path):
    """

    :param path:
    :return:
    """
    with open(path) as f:
        data = json.load(f)
        tokenizer = _tokenizer_from_json(data)

    return tokenizer


def store_tokenizer_to_path(tokenizer, path):
    """

    :param tokenizer: Tokenizer object to be stored
    :param path: designated path for it
    :return:
    """
    tokenizer_json = tokenizer.to_json()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


class TFImageLEDForConditionalGeneration(TFLEDForConditionalGeneration):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.image_encoder = tf.keras.applications.EfficientNetB1(
            include_top=False,
            weights="imagenet",
            pooling=None
        )

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs: Optional[TFLEDEncoderBaseModelOutput] = None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["labels"] is not None:
            inputs["use_cache"] = False
            if inputs["decoder_input_ids"] is None:
                inputs["decoder_input_ids"] = shift_tokens_right(
                    inputs["labels"], self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Encode the image
        if inputs["input_ids"] is not None:
            encoded_img = self.image_encoder(
                inputs["input_ids"], training=training)
            inputs["inputs_embeds"] = tf.reshape(
                encoded_img, (encoded_img.shape[0], -1, encoded_img.shape[-1]))
            inputs["input_ids"] = None

        outputs = self.led(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            encoder_outputs=inputs["encoder_outputs"],
            global_attention_mask=inputs["global_attention_mask"],
            head_mask=inputs["head_mask"],
            decoder_head_mask=inputs["decoder_head_mask"],
            past_key_values=inputs["past_key_values"],
            inputs_embeds=inputs["inputs_embeds"],
            decoder_inputs_embeds=inputs["decoder_inputs_embeds"],
            use_cache=inputs["use_cache"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        lm_logits = self.led.shared(outputs[0], mode="linear")
        lm_logits = lm_logits + self.final_logits_bias
        masked_lm_loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], lm_logits)

        if not inputs["return_dict"]:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return TFLEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
            encoder_global_attentions=outputs.encoder_global_attentions,
        )

    def generate(
        self,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
        use_cache=None,
        output_scores=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict_in_generate=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        **model_kwargs,
    ) -> Union[TFGreedySearchOutput, TFSampleOutput, TFBeamSearchOutput, TFBeamSampleOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code
        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it with
                :obj:`bos_token_id` and a batch size of 1.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.

                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

                `What are attention masks? <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            forced_bos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
                Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
                needs to be the target language token.
            forced_eos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the last generated token when :obj:`max_length` is reached.
            model_specific_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`tf.Tensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`tf.Tensor`.

                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.TFGreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.TFSampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSampleDecoderOnlyOutput`

                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.TFGreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.TFSampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSampleEncoderDecoderOutput`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from huggingface.co and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print(f'Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from huggingface.co and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print(f'Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from huggingface.co and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
            for i in range(3): #  3 output sequences were generated
                print(f'Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from huggingface.co and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print(f'Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from huggingface.co and cache.
            input_context = 'My cute dog'
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `TFOpenAIGPTLMHeadModel`, `TFXLNetLMHeadModel`, `TFGPT2LMHeadModel`, `TFCTRLLMHeadModel`, `TFT5ForConditionalGeneration`, `TFTransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_scores"] = output_scores
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        if self.config.is_encoder_decoder:
            model_kwargs["encoder_attentions"] = None
            model_kwargs["encoder_hidden_states"] = None

        if input_ids is not None:
            batch_size = shape_list(input_ids)[0]  # overridden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        inputs_embeds = model_kwargs.get("inputs_embeds")
        if input_ids is not None:
            # Encode the image
            encoded_img = self.image_encoder(
                input_ids, training=False)
            inputs_embeds = tf.reshape(
                encoded_img, (encoded_img.shape[0], -1, encoded_img.shape[-1]))
            input_ids = None

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        # if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids.numpy()):
        #     attention_mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32)
        # elif attention_mask is None:
        if attention_mask is None:
            attention_mask = tf.ones(inputs_embeds.shape[:-1])

        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to {eos_token_id} (first `eos_token_id`) to generate sequence")
            pad_token_id = eos_token_id

        # current position and vocab size
        # cur_len = shape_list(input_ids)[1]  # unused
        vocab_size = self.config.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), f"{self} should have a 'get_encoder' function defined"
            assert callable(self.get_encoder), f"{self.get_encoder} should be a method"

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_outputs = encoder(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if return_dict_in_generate:
                if output_attentions:
                    model_kwargs["encoder_attentions"] = encoder_outputs.attentions
                if output_hidden_states:
                    model_kwargs["encoder_hidden_states"] = encoder_outputs.hidden_states

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            raise NotImplementedError

            input_ids_len = shape_list(input_ids)[-1]
            input_ids = tf.broadcast_to(
                tf.expand_dims(input_ids, 1), (batch_size, effective_batch_mult * num_beams, input_ids_len)
            )
            attention_mask = tf.broadcast_to(
                tf.expand_dims(attention_mask, 1), (batch_size, effective_batch_mult * num_beams, input_ids_len)
            )
            input_ids = tf.reshape(
                input_ids, (effective_batch_size * num_beams, input_ids_len)
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = tf.reshape(
                attention_mask, (effective_batch_size * num_beams, input_ids_len)
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:

            # create empty decoder_input_ids
            input_ids = (
                tf.ones(
                    (effective_batch_size * num_beams, 1),
                    dtype=tf.int32,
                )
                * decoder_start_token_id
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = tf.reshape(
                tf.repeat(tf.expand_dims(tf.range(batch_size), -1), repeats=num_beams * effective_batch_mult, axis=1),
                shape=(-1,),
            )
            # expand encoder_outputs
            encoder_outputs = (tf.gather(encoder_outputs[0], expanded_batch_idxs, axis=0),)
        else:
            encoder_outputs = None
            cur_len = shape_list(input_ids)[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                forced_bos_token_id=forced_bos_token_id,
                forced_eos_token_id=forced_eos_token_id,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        return output
