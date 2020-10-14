"""
Functions related to dataset handling

Main function: convert the dataset to TFRecord File
"""

from common_definitions import *
from annotations import load_json_from_path
from html_SXN_parser.parser import encode_2_sxn


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_features(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_features(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_image(image_path):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_png(img, channels=3)
	img = tf.image.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
	img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
	return img

def get_images_dataset(annotations_path):
	"""
	Get only images as datasets. This function is designed for autoencoder in image_feature_extract.py

	:param annotations_path:
	:return: tf.data.Dataset
	"""

	annotations = load_json_from_path(annotations_path)

	# get the image ids only
	image_ids = [ann["image_id"]+".png" for ann in annotations]  # add with .png, because the files are in png

	# Feel free to change batch_size according to your system configuration
	image_dataset = tf.data.Dataset.from_tensor_slices(image_ids)
	image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	image_dataset = image_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE_AUTOENCODER)
	image_dataset = image_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return image_dataset

def convert_and_write_all_datasets(annotations_path, filename):
	"""
	Convert to TFRecord File so that later can be read to tf.data.Dataset
	:param filename: filename of the TFRecord File
	:param annotations_path:
	:return: Tokenizer object, max position of the chunk
	"""

	annotations = load_json_from_path(annotations_path)

	# get the image ids only
	image_ids = [ann["image_id"] for ann in annotations]  # add with .png, because the files are in png
	html_ids = [ann["html_id"] + ".html" for ann in annotations]  # add with .png, because the files are in png

	# get the sxns
	sxns = []
	for html_id in html_ids:
		with open(html_id, 'r') as f:
			sxn = encode_2_sxn(f.read())
			sxns.append("<start> " + sxn + " <end>")

	# tokenize the sxns
	tokenizer = tf.keras.preprocessing.text.Tokenizer(TOP_K, filters='', split=' ', oov_token="oov")
	tokenizer.fit_on_texts(sxns)

	tokenizer.word_index['<pad>'] = 0
	tokenizer.index_word[0] = '<pad>'

	seqs = tokenizer.texts_to_sequences(sxns)

	# constrain data to MAX_SEQ_LEN_DATASET
	positions = len(seqs) * [0]
	max_position = 0

	def append_tensors(j, i_seq, new_seq):
		return new_seq, j, image_ids[i_seq]

	for i_seq, seq in enumerate(seqs):
		len_seq = len(seq)
		if len_seq < MAX_SEQ_LEN_DATASET:
			seq = seq + (MAX_SEQ_LEN_DATASET - len_seq) * [0]
		elif len_seq > MAX_SEQ_LEN_DATASET:
			tensors = [append_tensors(j, i_seq, seq[j:MAX_SEQ_LEN_DATASET + j]) for j in range(1, len_seq - MAX_SEQ_LEN_DATASET + 1)]
			tensors = list(zip(*tensors))  # transpose it

			# put it in global variables
			seqs.extend(tensors[0])
			positions.extend(tensors[1])
			image_ids.extend(tensors[2])

			if (len_seq - MAX_SEQ_LEN_DATASET) > max_position:
				max_position = len_seq - MAX_SEQ_LEN_DATASET

			seq = seq[:MAX_SEQ_LEN_DATASET]  # replace seq with truncated one
		else:
			continue

		seqs[i_seq] = seq

	# Feel free to change batch_size according to your system configuration
	_datasets = tf.data.Dataset.from_tensor_slices((image_ids, seqs, positions))

	def serialize_example_pyfunction(feature0, feature1, feature2):
		"""
		Creates a tf.Example message ready to be written to a file.
		"""

		# Create a dictionary mapping the feature name to the tf.Example-compatible
		# data type.

		feature = {
			'image_id': _bytes_feature(feature0.numpy()),
			'sxn_token': _int64_features(feature1.numpy()),
			'pos': _int64_feature(feature2.numpy()),
		}

		# Create a Features message using tf.train.Example.

		example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
		return example_proto.SerializeToString()

	def tf_serialize_example(f0, f1, f2):
		tf_string = tf.py_function(
			serialize_example_pyfunction,
			(f0, f1, f2),  # pass these args to the above function.
			tf.string)  # the return type is `tf.string`.
		return tf.reshape(tf_string, ())  # The result is a scalar

	serialized_datasets = _datasets.map(tf_serialize_example)
	writer = tf.data.experimental.TFRecordWriter(filename)
	writer.write(serialized_datasets)

	return tokenizer, max_position


def get_all_datasets(filename):
	"""
	Get only images as datasets

	:param filename:
	:return: tf.data.Dataset
	"""
	filenames = [filename]
	raw_dataset = tf.data.TFRecordDataset(filenames)

	# Create a description of the features.
	feature_description = {
		'image_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
		'sxn_token': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),  # sxn in token
		'pos': tf.io.FixedLenFeature([], tf.int64, default_value=0),  # init position
	}

	def _parse_function(example_proto):
		# Parse the input tf.Example proto using the dictionary above.
		return tf.io.parse_single_example(example_proto, feature_description)

	def convert_data(dataset):
		return load_image(dataset["image_id"]+".png"), tf.cast(dataset["sxn_token"], tf.int32), tf.cast(dataset["pos"], tf.int32)

	_datasets = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # map the string to real data
	_datasets = _datasets.map(convert_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	# split datasets to train_datasets, and test_dataset... right now just set N_TEST_DATASET test_dataset
	_test_dataset = _datasets.take(N_TEST_DATASET)  # (N_TEST_DATASET, ...)
	_train_datasets = _datasets.skip(N_TEST_DATASET)

	_train_datasets = _train_datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE) # (:, :, ...)
	_train_datasets = _train_datasets.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return _train_datasets, _test_dataset  # (:, :, image, sxn_token, pos), (3, image, sxn_token, pos)

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

def store_additional_info(dict, filename):
	with open(filename, 'w') as outfile:
		json.dump(dict, outfile)

def load_additional_info(filename):
	with open(filename) as infile:
		data = json.load(infile)
	return data

if __name__ == "__main__":
	print("convert_and_write_all_datasets")
	tokenizer, max_position = convert_and_write_all_datasets(ANNOTATIONS_PATH, TFRECORD_FILENAME)  # convert and write all datasets from the annotations path to TFRecord File

	# store Tokenizer object to path
	print("store_tokenizer_to_path")
	store_tokenizer_to_path(tokenizer, TOKENIZER_FILENAME)

	# store additional data to path
	print("store_additional_info")
	store_additional_info({"max_pos": max_position}, ADDITIONAL_FILENAME)