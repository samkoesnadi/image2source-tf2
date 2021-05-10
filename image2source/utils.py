"""
Utils function that can be used as auxiliary
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf

from .common_definitions import *


def pairs_from_parent_path(parent_path):
	"""
	make function to list id from parent path
	:param parent_path: the path that consists all of the files
	:return: dictionary of html_ids and image_ids
	"""
	annotations = []
	ids = [f[:-4] for f in os.listdir(parent_path)   # ids that available in the PATH
		   if os.path.isfile(os.path.join(parent_path, f)) and f[-4:] == ".png"]
	if len(ids) == 0:
		logging.warning("There is no ids found in the file")
	for _id in ids:
		_id = os.path.join(parent_path, _id)
		annotations.append({"html_id": _id, "image_id": _id})
	return annotations

def dump_json_to_path(path, _arr):
	"""

	:param path:
	:param _arr:
	:return: None
	"""
	with open(path, 'w') as f:
		json.dump(_arr, f)
		logging.debug("JSON is written to %s", path)

def load_json_from_path(path):
	"""

	:param path:
	:return:
	"""
	annotations = []
	with open(path, 'r') as f:
		annotations = json.load(f)
		logging.debug("JSON is read from %s", path)
	return annotations

def save_fig_png(input_arr, filename):
	"""

	:param filename:
	:param input_arr: (batch, height, width, channel)
	:return:
	"""

	input_arr = input_arr[0]  # pick the first batch
	input_arr = np.transpose(input_arr, (2,0,1))

	fig = plt.figure(figsize=(10, 10))
	len_arr = len(input_arr)

	for i, inp in enumerate(input_arr):
		ax = fig.add_subplot(math.ceil(len_arr**.5), math.ceil(len_arr**.5), i+1)  # total_x?, total_y?, index
		ax.set_title(str(inp.min())+","+str(inp.max()))  # title is min max value
		ax.imshow(inp)

	plt.savefig("layers_figure/"+filename+".png", bbox_inches="tight")
	plt.close()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # this one will go down second time some how
	def __init__(self, d_model, warmup_steps=4000, multiplier=1.):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps
		self.multiplier = multiplier

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step) / tf.maximum((step - self.warmup_steps) * self.multiplier / (self.warmup_steps * 2), 1)
		# arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def weighted_loss(target, pred, loss_function, light_background=True):
	"""

	:param target:
	:param pred:
	:param loss_function: higher-order loss function
	:param light_background: Boolean if darker produces more attention
	:return:
	"""

	if loss_function == tf.keras.losses.MeanSquaredError:
		_loss = tf.keras.losses.MeanSquaredError(reduction="none")(target, pred)
	else:
		raise Exception("Loss function is not recognized by weighted_loss function")

	# weight the loss. TODO: change it if you want to change the ratio. now, darker thing means higher weight
	avg_pred = tf.math.reduce_mean(pred, -1)
	min_val = tf.math.reduce_min(avg_pred)
	max_val = tf.math.reduce_max(avg_pred)

	if light_background:
		ratio = (1 - (avg_pred - min_val) / (max_val - min_val)) + 1  # minus one because lighter color is less attention
	else:
		ratio = (avg_pred - min_val) / (max_val - min_val) + 1

	# _loss = tf.math.reduce_sum(ratio * _loss) / tf.math.reduce_sum(ratio)
	_loss = tf.math.reduce_sum(ratio * _loss)

	return _loss


class SmartCheckpointSaver:
	def __init__(self, ckpt_manager):
		self.ckpt_manager = ckpt_manager
		self.max_val_acc = -np.inf  # max validation accuracy
		self.max_acc_epoch = 0  # the epoch in which we have the maximum accuracy

	def __call__(self, curr_epoch, curr_val_acc):
		"""

		:param curr_epoch:
		:param curr_val_acc:
		:return: 1 ckpt saved, 0 nothing is done, -1 no new max_val_acc is created in the given rule
		"""

		# just for beginning when max_val and max_acc is empty
		if self.max_acc_epoch == 0:
			self.max_val_acc = curr_val_acc
			self.max_acc_epoch = curr_epoch

		if curr_val_acc > self.max_val_acc:
			ckpt_save_path = self.ckpt_manager.save()
			print('Saving checkpoint for epoch {} at {}'.format(curr_epoch,
			                                                    ckpt_save_path))
			self.max_val_acc = curr_val_acc
			self.max_acc_epoch = curr_epoch
			return 1
		elif curr_epoch <= MIN_EPOCH_TO_BREAK:  # if it is less or equal to MIN_EPOCH_TO_BREAK, reset everything
			self.max_val_acc = curr_val_acc
			self.max_acc_epoch = curr_epoch
		else:
			epoch_min = min(EPOCHS, max(MIN_EPOCH_TO_BREAK, int(self.max_acc_epoch * 2.)), int(self.max_acc_epoch + GAP_OF_DEAD_EPOCH))  # min epoch to break is 10

			if epoch_min <= curr_epoch:
				return -1
		return 0


class FocalLoss:
	def __init__(self, ):
		self.step = 0

	def __call__(self, target_tensor, prediction_tensor):

		# update dynamic Focal Loss
		# # this is polynomial function
		# alpha = max((ALPHA_BALANCED - .75) * (self.step / (WARM_UP_STEPS // 2)) ** 2 + .75, ALPHA_BALANCED)
		# gamma = min((GAMMA_FOCAL - 0.) * (self.step / (WARM_UP_STEPS // 2)) ** 2 + 0., GAMMA_FOCAL)

		alpha = ALPHA_BALANCED
		gamma = GAMMA_FOCAL

		if SIGMOID:
			sigmoid_p = tf.nn.sigmoid(prediction_tensor)
		else:
			sigmoid_p = tf.nn.softmax(prediction_tensor)
		sigmoid_p = tf.clip_by_value(sigmoid_p, MIN_EPSILON, MAX_EPSILON)  # clip the value
		zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

		# For poitive prediction, only need consider front part loss, back part is 0;
		# target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
		pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

		# For negative prediction, only need consider back part loss, front part is 0;
		# target_tensor > zeros <=> z=1, so negative coefficient = 0.
		neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)
		per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(sigmoid_p) \
		                      - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(1.0 - sigmoid_p)

		# update internal step
		self.step += 1

		return tf.reduce_sum(per_entry_cross_ent, axis=-1)
