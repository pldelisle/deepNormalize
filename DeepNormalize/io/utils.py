import nibabel as nib
import re
import os
import numpy as np
from skimage.util import view_as_blocks, view_as_windows
from DeepNormalize.utils.nilearn_utils import crop_img_to


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def read_image_files(image_files, crop=None):
    image_list = list()
    for index, image_file in enumerate(image_files):
        image_list.append(read_image(image_file, crop=crop))

    return image_list


def read_image(in_file, crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file.decode()))
    # image = nib.load(in_file)
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    return image


def extract_patches(image, window_shape=(40, 40, 40), step=10):
    patch_image = view_as_windows(image, window_shape=window_shape, step=step)
    return patch_image


def extract_blocks(image, block_shape=(40, 40, 40)):
    block_image = view_as_blocks(image, block_shape=block_shape)
    return block_image


def pad_image_and_center(image):
    centered_image = np.zeros((240, 240, 160), dtype=np.int64)
    d = int(np.floor(np.abs(centered_image.shape[0] - image.shape[0])) / 2)
    e = int(np.floor(np.abs(centered_image.shape[1] - image.shape[1])) / 2)
    f = int(np.floor(np.abs(centered_image.shape[2] - image.shape[2])) / 2)
    centered_image[d:d + image.shape[0], e:e + image.shape[1], f:f + image.shape[2]] = image

    return centered_image


def load_image_obj(filename):
    try:
        image = nib.load(filename)
        image_data = image.get_data().astype(np.int64)
        return image_data
    except nib.filebasedimages.ImageFileError:
        raise IOError('Nibabel could not load image file: {}'.format(filename))


def _get_file_paths(path):
    file_paths = list()
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            f = os.path.join(dirpath, name)
            file_paths.append(f)

        file_paths = list(filter(None, file_paths))

    return file_paths


def find_files(path, modalities):
    files = list()

    for modality in modalities:
        modality = modality.lower()
        regex = re.compile(r"" + modality + r".nii")

        file_paths = _get_file_paths(path)

        modality_files = list(filter(regex.search, file_paths))

        files = files + modality_files

    return files


def find_segmentations(path):
    file_paths = _get_file_paths(path)
    regex_seg = re.compile(r"seg.nii")
    segmentation_files = list(filter(regex_seg.search, file_paths))

    return segmentation_files


def load_dataset(files):
    data_set = list()

    for file in files:
        image = load_image_obj(file)
        data_set.append(image)

    return data_set


