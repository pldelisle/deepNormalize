import numpy as np

from nilearn.image import new_img_like
from nilearn.image import crop_img
from DeepNormalize.io.utils import read_image, read_image_files

TOLERANCE = 1e-5
BACKGROUND_VALUE = 0


def preprocess_images(data):
	"""
	Preprocess a list of images. Each first-layer list represents a modality, while
	each modality contains a list of "batch_size" images, each belonging to a subject.
	Modalities in order: FLAIR, T1, T1ce, T2, Segmentation, Weight Map.
	:param data: The image list, each entry representing a subject.
	"""

	# Get each modality.
	flair = data[0]
	t1 = data[1]
	t1ce = data[2]
	t2 = data[3]
	segmentation = data[4]
	weight_map = data[5]

	# Get the T1 modality slices.
	# Template used for each subject is the T1 modality (current_data[1])
	t1_slices = get_slices(t1)

	# Get the cropped modalities.
	cropped_t1 = t1[tuple(t1_slices)]
	cropped_flair = flair[tuple(t1_slices)]
	cropped_t1ce = t1ce[tuple(t1_slices)]
	cropped_t2 = t2[tuple(t1_slices)]
	cropped_segmentation = segmentation[tuple(t1_slices)]
	cropped_weight_map = weight_map[tuple(t1_slices)]

	# Additional preprocessing here.
	# ...

	return cropped_t1, cropped_flair, cropped_t1ce, cropped_t2, cropped_segmentation, cropped_weight_map


def get_slices(volume_data):
	"""
	Return a set of slices to crop.
	:param volume_data: a Numpy array containing voxel values.
	:return: cropped_data : a Numpy array containing slices that defines the range of the crop.
	E.g. [slice(20, 200), slice(40, 150), slice(0, 100)] defines a 3D cube
	"""
	# Logical OR to find non-background values. True if voxel value is below or above 0.
	idx = np.logical_or(volume_data < (BACKGROUND_VALUE - TOLERANCE),
						volume_data > (BACKGROUND_VALUE + TOLERANCE))

	passes_threshold = np.any(idx, axis=-1)
	coords = np.array(np.where(passes_threshold))
	start = coords.min(axis=1)
	end = coords.max(axis=1) + 1

	# Pad with one voxel to avoid resampling problems.
	start = np.maximum(start - 1, 0)
	end = np.minimum(end + 1, volume_data.shape[:3])

	# Create slices of the volume.
	slices = [slice(s, e) for s, e in zip(start, end)]

	return slices


def reslice_image_set(set_of_files, segmentations, crop=True):
	if crop:
		crop_slices = get_cropping_parameters(set_of_files)
	else:
		crop_slices = None

	images = read_image_files(set_of_files, crop=crop_slices)
	segs = read_image_files(segmentations, crop=crop_slices)

	return images, segs


def get_cropping_parameters(in_files):
	foreground = get_complete_foreground(in_files)
	return crop_img(foreground, return_slices=True, copy=True)


def get_complete_foreground(training_data_files):
	subject_foreground = get_foreground_from_set_of_files(training_data_files)

	return new_img_like(read_image(training_data_files[0]), subject_foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001):
	for i, image_file in enumerate(set_of_files):
		image = read_image(image_file)
		is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
									  image.get_data() > (background_value + tolerance))
		if i == 0:
			foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

		foreground[is_foreground] = 1

	return foreground
