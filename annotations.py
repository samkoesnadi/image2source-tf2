"""
Select and list all the datasets are wanted for the project. They would be stored in a dictionary of html_id and image_id. This dictionary is stored in json file.
Author: Samuel Koesnadi 2019
"""

import os
import json
from common_definitions import *

def pairs_from_parent_path(parent_path):
	"""
	make function to list id from parent path
	:param parent_path: the path that consists all of the files
	:return: dictionary of html_ids and image_ids
	"""
	annotations = []
	ids = [f[:-4] for f in os.listdir(parent_path) if os.path.isfile(os.path.join(parent_path, f)) and f[-4:] == ".png"]  # ids that available in the PATH
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


if __name__ == "__main__":
	### annotate pix2code's datasets
	PATH = "../datasets/pix2code"
	dump_json_to_path(ANNOTATIONS_PATH, pairs_from_parent_path(PATH))
