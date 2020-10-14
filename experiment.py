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

a = tf.math.top_k([[1,2,3, 4, 5, 6, 7, 8, 9],[4,5,6,10,11,12,13,14,15]], BEAM_SEARCH_N)
print(a)


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

class SCSTCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # this one will go down second time some how
	def __init__(self, max_temperature, min_temperature):
		super(SCSTCustomSchedule, self).__init__()

		if max_temperature < min_temperature:
			raise Exception("max temperature is less than min temperature")

		self.max_temperature = max_temperature
		self.min_temperature = min_temperature

		self.multiplier = 10 ** -4.7 if self.max_temperature >= 1 else 10 ** 1.6


	def __call__(self, step):
		return tf.maximum(self.min_temperature, self.max_temperature * tf.exp(-self.max_temperature * self.multiplier * step))

temp_learning_rate_schedule = CustomSchedule(dff, WARM_UP_STEPS, .3)
# a = tf.random.normal([100])
# a = tf.nn.softmax(a/5)
# temp_learning_rate_schedule = SCSTCustomSchedule(5e-6, 5e-7)
# temp_learning_rate_schedule = lambda t : tf.minimum(1e-4*t, 3e-4)
print(tf.reduce_max(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32))))
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.plot(a)
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()