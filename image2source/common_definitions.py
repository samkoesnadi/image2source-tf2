
d_model = 1280  # this has to match with the image encoder

TARGET_SIZE = 320
encoder_max_length = 100
decoder_max_length = 1536
tokenizer_max_vocab_size = 2700

led_partial_configs = {
    "vocab_size": tokenizer_max_vocab_size,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "decoder_ffn_dim": 2048,
    "encoder_ffn_dim": 2048,
    "max_encoder_position_embeddings": 256,
    "max_decoder_position_embeddings": 2048,
    "d_model": d_model,
    "attention_window": [
        encoder_max_length,
        encoder_max_length
    ],

    "dropout": 0.25,
    "attention_dropout": 0.25,
    "activation_dropout": 0.25,
    "classifier_dropout": 0.25,
    "classif_dropout": 0.25
}
