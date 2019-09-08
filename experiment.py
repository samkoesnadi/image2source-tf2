"""
Scratch page that you can experiment with anything you want
Author: Samuel Koesnadi 2019
"""

from common_definitions import *
from html_SXN_parser.parser import encode_2_sxn, decode_2_html

# experiment with tokenizing SXN
with open("html_SXN_parser/generated_example.html", "r") as f:
	html = f.read()
sxn = encode_2_sxn(html)
print(sxn)

tokenizer = tf.keras.preprocessing.text.Tokenizer(TOP_K, filters='', split=' ', oov_token="<unk>")
tokenizer.fit_on_texts([sxn])
train_seqs = tokenizer.texts_to_sequences([sxn])
print(train_seqs)
print(list(map(lambda i: tokenizer.index_word[i], train_seqs[0])))
print(tokenizer.sequences_to_texts(train_seqs))
print(decode_2_html(tokenizer.sequences_to_texts(train_seqs)[0]))