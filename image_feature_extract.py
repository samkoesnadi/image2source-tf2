"""
Extract image's feature by utilizing MobileNetV2 (This net can be changed). To train the network, decoder is built on top of the MobileNetV2 to create a whole autoencoder
Author: Samuel Koesnadi 2019
"""

from common_definitions import *
from dataset import get_images_dataset, load_additional_info, store_additional_info
from utils import *

### Create Model
class MobileNetV2_AutoEncoder():
	def __init__(self, checkpoint_path):
		"""

		:param checkpoint_path: path to checkpoint for training and evaluation
		"""
		# Encoder
		image_input = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE,IMAGE_INPUT_SIZE,3,))
		self.encoder = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
		                                                weights=None, input_tensor=image_input)

		self.hidden_layer = self.encoder.layers[-1].output

		# Decoder
		dec_hidden_4 = self.decoder_block(self.hidden_layer, [320, 160])

		dec_hidden_3 = self.decoder_block(dec_hidden_4, [96, 64])

		dec_hidden_2 = self.decoder_block_bottleneck(dec_hidden_3, [32, 28])

		dec_hidden_1 = self.decoder_block_bottleneck(dec_hidden_2, [24, 20])

		output = self.decoder_block_bottleneck(dec_hidden_1, [16, 12])

		output = tf.keras.layers.Conv2D(8, 3, padding="same", activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(output)
		output = tf.keras.layers.BatchNormalization()(output)

		# Final CNN Layer with linear activation
		self.output = tf.keras.layers.Conv2D(3, 3, padding="same", kernel_initializer=KERNEL_INITIALIZER, activation="linear")(output)

		# Model definition
		self.model = tf.keras.Model(image_input, self.output)
		self.model_evaluate = tf.keras.Model(image_input, [self.output, dec_hidden_4, dec_hidden_3, dec_hidden_2, dec_hidden_1, output])

		# define optimizer and losses
		self.optimizer = tf.keras.optimizers.Adam(DEFAULT_LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

		# for additional information
		self.train_loss = tf.keras.metrics.Mean(name="mobilenet_train_loss")

		# set checkpoint
		self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
		self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)

		self.smart_ckpt_saver = SmartCheckpointSaver(self.ckpt_manager)

		# if a checkpoint exists, restore the latest checkpoint.
		if self.ckpt_manager.latest_checkpoint:
			self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
			print('Latest checkpoint restored from', checkpoint_path, "!!!")

	def loss(self, target, pred):
		# return weighted_loss(target, pred, tf.keras.losses.MeanSquaredError)
		return tf.reduce_sum(tf.keras.losses.MeanSquaredError(reduction="none")(target, pred))

	def decoder_block(self, inp, filters_arr):

		"""
		Block for each step in the decoder
		:param inp: input or intermediate layer
		:param filters_arr: array of filters for the CNN (Conv, conv, deconv and conv)
		:return: tf.keras.layers.conv2D
		"""
		x = tf.keras.layers.Conv2D(filters_arr[0], 3, padding="same", activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(inp)
		x = tf.keras.layers.Conv2D(filters_arr[1], 3, padding="same",  activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.UpSampling2D()(x)

		return x

	def decoder_block_bottleneck(self, inp, filters_arr):

		"""
		Block for each step in the decoder
		:param inp: input or intermediate layer
		:param filters_arr: array of filters for the CNN (Conv, conv, deconv and conv)
		:return: tf.keras.layers.conv2D
		"""
		x = tf.keras.layers.Conv2D(filters_arr[0], 3, padding="same", activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(inp)
		x = tf.keras.layers.Conv2D(filters_arr[0] + (filters_arr[1] - filters_arr[0]), 1, padding="same",  activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(x)
		x = tf.keras.layers.Conv2D(filters_arr[1], 3, padding="same",  activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.UpSampling2D()(x)

		return x

	### Training pipeline
	@tf.function
	def train_step(self, img_tensor, target):
		with tf.GradientTape() as tape:
			prediction = self.model(img_tensor)
			loss = self.loss(target, prediction)

		trainable_variables = self.model.trainable_variables
		gradients = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))  # minimize loss

		self.train_loss(loss)


	def evaluate(self, img_tensor, target):
		global IS_TRAINING

		# change IS_TRAINING to False
		temp_is_training = IS_TRAINING
		IS_TRAINING = False

		all_vars = self.model_evaluate(img_tensor)
		prediction = all_vars[0]

		# store the figures
		for i, layer in enumerate(all_vars):
			save_fig_png(layer, "mobilnetv2_autoencoder/prediction_image_feature_layer_"+str(i))

		loss = self.loss(target, prediction)
		print("The calculated loss results {}".format(loss))

		fig = plt.figure(figsize=(10, 10))
		ax_target = fig.add_subplot(1, 2, 1)  # total_x?, total_y?, index
		ax_target.set_title("Target")
		ax_target.imshow(img_tensor[-1])

		ax_pred = fig.add_subplot(1, 2, 2)  # total_x?, total_y?, index
		ax_pred.set_title("Prediction l="+str(loss))
		ax_pred.imshow(prediction[-1])

		plt.savefig("Image_feature_result.png", bbox_inches="tight")

		IS_TRAINING = temp_is_training

if __name__ == "__main__":
	autoencoder = MobileNetV2_AutoEncoder(AUTOENCODER_CHECKPOINT_PATH)

	### Handle Dataset
	train_dataset = get_images_dataset(ANNOTATIONS_PATH)
	additional_info = load_additional_info(ADDITIONAL_FILENAME)

	if IS_TRAINING:
		### Train loop
		start_epoch = 0
		if autoencoder.ckpt_manager.latest_checkpoint:
			start_epoch = additional_info["autoencoder_epoch"]

		for epoch in range(start_epoch, EPOCHS):
			start = time.time()

			autoencoder.train_loss.reset_states()

			for (batch, img_tensor) in enumerate(train_dataset):
				autoencoder.train_step(img_tensor, img_tensor)  # same input and target because it is a freakin autoencoder

				if batch % 100 == 0:
					print('Epoch {} Batch {} Loss {:.4f}'.format(
						epoch + 1, batch, autoencoder.train_loss.result()))

			if (epoch + 1) % 5 == 0:
				if (epoch + 1) <= 10 or (epoch + 1) % 50 == 0:
					autoencoder.evaluate(img_tensor, img_tensor)

			print('Epoch {} Loss {:.4f}'.format(epoch + 1, autoencoder.train_loss.result()))
			print('Time taken for 1 epoch: {} secs'.format(time.time() - start))

			additional_info["autoencoder_epoch"] = epoch
			store_additional_info(additional_info, ADDITIONAL_FILENAME)

			should_break = autoencoder.smart_ckpt_saver(epoch+1, -autoencoder.train_loss.result())  # minus it because we do not have real accuracy, so just make a virtual one by minusing it
			if should_break == -1:
				break

			print()

		print ("Training over...")

		print('Saving MobileNetV2 weights for epoch {}'.format(autoencoder.smart_ckpt_saver.max_acc_epoch))
		autoencoder.ckpt.restore(autoencoder.ckpt_manager.latest_checkpoint)  # load checkpoint that was just trained to model
		autoencoder.encoder.save_weights(MOBILENETV2_WEIGHT_PATH)  # save the preprocessing weights

	print ("Start evaluation...")
	eval_dataset = next(iter(train_dataset))  # TODO: this definitely needs to be changed with more proper pipeline
	autoencoder.evaluate(eval_dataset, eval_dataset)