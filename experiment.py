"""
Scratch page that you can experiment with anything you want
Author: Samuel Koesnadi 2019
"""
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")



from utils import *
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

temp_learning_rate_schedule = CustomSchedule(2094, WARM_UP_STEPS)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()