import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
import nibabel as nib
import argparse
import os


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def generate_ROI(volume):
	"""
	Returns a region of interest (ROI) map where voxel intensity is higher than 0.
	:param volume: A 3D image.
	:return: A 3D array containing the region of interest (1 if condition is verified).
	"""
	idx = np.where(volume > 0)

	roiVolume = np.zeros(volume.shape, dtype=np.int64)

	roiVolume[idx] = 1

	return roiVolume


def construct_weights_and_mask(volume):
	"""
	Constructs a weight map.
	:param volume: A 3D volume (segmentation).
	:return: A 3D weight map of the contours of an object.
	"""

	W_0, SIGMA = 10, 5
	seg_boundaries = find_boundaries(volume, mode='inner')

	bin_img = volume > 0
	# take segmentations, ignore boundaries
	binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

	foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
	background_weight = 1 - foreground_weight

	# build euclidean distances maps for each cell:
	cell_ids = [x for x in np.unique(volume) if x > 0]
	distances = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], len(cell_ids)))

	for i, cell_id in enumerate(cell_ids):
		distances[..., i] = distance_transform_edt(volume != cell_id)

	# we need to look at the two smallest distances
	distances.sort(axis=-1)

	weight_map = W_0 * np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))
	weight_map[binary_with_borders] = foreground_weight
	weight_map[~binary_with_borders] += background_weight

	return weight_map, binary_with_borders


def create_file(X, y, filename):
	"""
	Create a TFRecord file.
	:param X: A list containing all file paths to be written.
	:param y: A list containing segmentation file paths to be written.
	:param filename: The file name of the TFRecords to be written.
	:return:
	"""
	writer = tf.python_io.TFRecordWriter(filename)

	for i in range(len(X)):  # For all subjects.

		print("Processing subject " + str(i + 1) + " of " + str(len(X)))

		modalities = list()

		for j in range(len(X[i])):  # For all subject's modalities, load the image.
			modalities.append(nib.load(X[i][j]).get_data().astype(np.int64))

		# Load the segmentation of the current subject i.
		seg = nib.load(y[i]).get_data().astype(np.int64)

		# Make all classes contiguous in [0, 3] space. There is no label==3 in BraTS data set.
		seg[seg == 4] = 3

		# Construct the weight map according to the segmentation.
		weight_map, _ = construct_weights_and_mask(seg)

		# Construct a TFRecord feature.
		feature = {
			"flair": _int_feature(modalities[0].ravel()),
			"t1": _int_feature(modalities[1].ravel()),
			"t1ce": _int_feature(modalities[2].ravel()),
			"t2": _int_feature((modalities[3]).ravel()),
			"segmentation": _int_feature(seg.ravel()),
			"weight_map": _float_feature(weight_map.ravel())
		}

		# Construct a TFRecord example.
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Write the example to file.
		writer.write(example.SerializeToString())

	writer.close()


def get_subjects_and_segmentations(path):
	"""
	Utility method to get, filter and arrange BraTS data set in a series of lists.
	:param path: The path to look for BraTS data set files.
	:return: A tuple containing multimodal MRI images for each subject and their respective segmentation.
	"""
	subjects = list()
	segs = list()

	for (dirpath, dirnames, filenames) in os.walk(path):
		if len(filenames) is not 0:
			segmentations = list(filter(lambda k: "seg.nii.gz" in k, filenames))
			files = filenames
			if segmentations:
				for seg in segmentations:
					files.remove(seg)
					segs.append(os.path.join(dirpath, seg))

			paths = list()
			for file in files:
				file = os.path.join(dirpath, file)
				paths.append(file)

			paths.sort()
			subjects.append(paths)

	del subjects[0]  # Remove the survival.csv file.

	return subjects, segs


def write_lists(filename, list_to_log, path):
	"""
	Write down to a file the elements of a list.
	:param filename: Output file name.
	:param list_to_log: The list to output to a text file.
	:param path: Path where to output the text file.
	:return: None.
	"""
	with open(os.path.join(path, filename), "w") as file:
		for item in list_to_log:
			file.write("%s\n" % item)
		file.close()


def main(args):
	subjects, segs = get_subjects_and_segmentations(args.data_dir)

	X_train, X_valid, y_train, y_valid = train_test_split(subjects, segs, test_size=0.10, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

	if not args.full:  # Create a small data set for testing purpose.
		X_train = X_train[:10]
		y_train = y_train[:10]

		X_test = X_test[:5]
		y_test = y_test[:5]

		X_valid = X_valid[:5]
		y_valid = y_valid[:5]

	print("Training set contains " + str(len(X_train)) + " images.")
	print("Validation set contains " + str(len(X_valid)) + " images.")
	print("Evaluation set contains " + str(len(X_test)) + " images.")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Writes to text files the content of each training, validation and test list.
	log_files = ["train_log.txt", "validation_log.txt", "test_log.txt"]
	lists = [X_train, X_valid, X_test]

	for filename, list_to_log in zip(log_files, lists):
		write_lists(filename, list_to_log, args.output_dir)

	# Create TFRecords files.
	training_file = args.output_dir + "/train.tfrecords"
	validation_file = args.output_dir + "/validation.tfrecords"
	test_file = args.output_dir + "/test.tfrecords"

	print("Processing training set.")
	create_file(X_train, y_train, training_file)

	print("Processing validation set.")
	create_file(X_valid, y_valid, validation_file)

	print("Processing test set.")
	create_file(X_test, y_test, test_file)

	return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--full',
		type=bool,
		required=True,
		help='Whether to process full data set or create a small one.')
	parser.add_argument(
		'--data-dir',
		type=str,
		required=True,
		help='The directory where the BraTS input data is stored.')
	parser.add_argument(
		'--output-dir',
		type=str,
		required=True,
		help='The directory where the output data is stored.')
	args = parser.parse_args()

	main(args=args)
