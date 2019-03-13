# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Generate TFRecords files for medical images.

"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt
from DeepNormalize.old.preprocessing import preprocess_images, get_patches
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

    # Take segmentations, ignore boundaries
    binary_with_borders = np.bitwise_xor(bin_img, seg_boundaries)

    foreground_weight = 1 - binary_with_borders.sum() / binary_with_borders.size
    background_weight = 1 - foreground_weight

    # Build euclidean distances maps for each cell:
    cell_ids = [x for x in np.unique(volume) if x > 0]
    distances = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], len(cell_ids)))

    for i, cell_id in enumerate(cell_ids):
        distances[..., i] = distance_transform_edt(volume != cell_id)

    # We need to look at the two smallest distances
    distances.sort(axis=-1)

    weight_map = W_0 * np.exp(-(1 / (2 * SIGMA ** 2)) * ((distances[..., 0] + distances[..., 1]) ** 2))
    weight_map[binary_with_borders] = foreground_weight
    weight_map[~binary_with_borders] += background_weight

    return weight_map, binary_with_borders


def write_training_examples(X, y, filename):
    """
    Create a TFRecord file for training phase (training and validation).
    :param X: A list containing all file paths to be written.
    :param y: A list containing segmentation file paths to be written.
    :param filename: The file name of the TFRecords to be written.
    :return:
    """

    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(len(X)):  # For all subjects.

        print("Processing subject " + str(i + 1) + " of " + str(len(X)))

        modalities = dict()
        modalities_name = ["flair", "t1", "t1ce", "t2"]

        for j, modality_name in zip(range(len(X[i])), modalities_name):  # For all subject's modalities, read file.
            # Loads the image.
            modality = nib.load(X[i][j]).get_data().astype(np.int64)
            # Expand one dimension. Will now get [H, W, D, 1] shape for current modality.
            modality = np.expand_dims(modality, axis=-1)
            # Append the current modality to a dictionary of modalities.
            modalities[modality_name] = modality

        # Load the segmentation of the current subject i.
        seg = nib.load(y[i]).get_data().astype(np.int64)

        # Make all classes contiguous in [0, 3] space. There is no label==3 in BraTS data set.
        seg[seg == 4] = 3

        # Construct the weight map according to the segmentation.
        weight_map, _ = construct_weights_and_mask(seg)

        # Expand one dimension. Will now get [H, W, D, 1] shape for segmentation.
        seg = np.expand_dims(seg, axis=-1)

        # Append segmentation to modality list.
        modalities["segmentation"] = seg

        # Expand one dimension. Will now get [H, W, D, 1] shape for weight map.
        weight_map = np.expand_dims(weight_map, axis=-1)

        # Append weight map to modality list.
        modalities["weight_map"] = weight_map

        # Get slices from preprocessing without applying crop.
        slices = preprocess_images(modalities, apply=False)

        # Get original and modified image shape.
        original_shape = [seg.shape[0], seg.shape[1], seg.shape[2], seg.shape[3]]

        # [X_start, X_step, X_stop, Y_start, Y_step, Y_stop, Z_start, Z_step, Z_stop]
        tf_slices = [slices[0].start, slices[0].step, slices[0].stop,
                     slices[1].start, slices[1].step, slices[1].stop,
                     slices[2].start, slices[2].step, slices[2].stop]

        # Construct a TFRecord feature.
        feature = {
            "flair": _int_feature(modalities["flair"].ravel()),
            "t1": _int_feature(modalities["t1"].ravel()),
            "t1ce": _int_feature(modalities["t1ce"].ravel()),
            "t2": _int_feature((modalities["t2"]).ravel()),
            "segmentation": _int_feature(modalities["segmentation"].ravel()),
            "weight_map": _float_feature(modalities["weight_map"].ravel()),
            "original_shape": _int_feature(original_shape),
            "slices": _int_feature(tf_slices)
        }

        # Construct a TFRecord example.
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Write the example to file.
        writer.write(example.SerializeToString())

    writer.close()


def write_testing_examples(X, y, output_dir, patch_shape, extraction_step):
    """
    Create a TFRecord file for testing phase.
    :param X: A list containing all file paths to be written.
    :param y: A list containing segmentation file paths to be written.
    :param patch_shape: Integer or tuple of length arr.ndim
                        Indicates the shape of the patches to be extracted. If an
                        integer is given, the shape will be a hypercube of
                        sidelength given by its value.
    :param extraction_step : Integer or tuple of length arr.ndim
                             Indicates step size at which extraction shall be performed.
                             If integer is given, then the step is uniform in all dimensions.
    :return: None
    """

    for i in range(len(X)):  # For all test subjects.

        filename = os.path.splitext(os.path.splitext(os.path.basename(X[i][0]))[0])[0]
        filename = filename.replace("_flair", "")
        test_file = "/test-" + filename + ".tfrecords"

        print("Processing subject " + str(i + 1) + " of " + str(len(X)) + " with file name " + output_dir + test_file)

        writer = tf.python_io.TFRecordWriter(output_dir + test_file)

        modalities = dict()
        modalities_name = ["flair", "t1", "t1ce", "t2"]

        for j, modality_name in zip(range(len(X[i])), modalities_name):  # For all subject's modalities.
            # Loads the image.
            modality = nib.load(X[i][j]).get_data().astype(np.int64)
            # Expand one dimension. Will now get [H, W, D, 1] shape.
            modality = np.expand_dims(modality, axis=-1)
            # Append the current modality to a dictionary of modalities.
            modalities[modality_name] = modality

        # Load the segmentation of the current subject i.
        seg = nib.load(y[i]).get_data().astype(np.int64)

        # Expand one dimension. Will now get [H, W, D, 1] shape for segmentation.
        seg = np.expand_dims(seg, axis=-1)

        # Append segmentation to modality list.
        modalities["segmentation"] = seg

        # Apply preprocessing.
        slices, modalities = preprocess_images(modalities, apply=True)

        # Get patches for all modalities. Give a [N_patches, patch_shape, patch_shape, patch_shape, 1] list for each
        # modality.
        modalities = get_patches(modalities, patch_shape=patch_shape, extraction_step=extraction_step)

        for k in range(0, modalities["t1"].shape[0]):  # Take the first modality for counting number of patches.
            # For each patch, create a feature containing all modalities.
            feature = {
                "flair": _int_feature(modalities["flair"][k].ravel()),
                "t1": _int_feature(modalities["t1"][k].ravel()),
                "t1ce": _int_feature(modalities["t1ce"][k].ravel()),
                "t2": _int_feature(modalities["t2"][k].ravel()),
                "segmentation": _int_feature(modalities["segmentation"][k].ravel()),
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

    if args.full == "False":  # Create a small data set for testing purpose.
        X_train = X_train[:10]
        y_train = y_train[:10]

        X_valid = X_valid[:5]
        y_valid = y_valid[:5]

        X_test = X_test[:5]
        y_test = y_test[:5]

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

    patch_shape = [args.patch_shape, args.patch_shape, args.patch_shape, 1]

    print("Processing training set.")
    write_training_examples(X_train, y_train, training_file)

    print("Processing validation set.")
    write_training_examples(X_valid, y_valid, validation_file)

    print("Processing test set.")
    write_testing_examples(X_test, y_test, args.output_dir, patch_shape, args.extraction_step)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--full',
        default=True,
        required=True,
        help='Whether to process full data set or create a small one.')
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='The directory where the BraTS input data is stored.')
    parser.add_argument(
        '--patch-shape',
        type=int,
        required=True,
        help='The size of the desired 3D patch for testing TFRecord file.')
    parser.add_argument(
        '--extraction-step',
        type=int,
        required=True,
        help='Indicates step size at which patch extraction shall be performed.')
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='The directory where the output data is stored.')
    args = parser.parse_args()

    main(args=args)
