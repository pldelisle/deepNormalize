import tensorflow as tf
import os
import glob
import numpy as np
import nibabel as nib

from DeepNormalize.utils import preprocessing

from sklearn.model_selection import train_test_split

HEIGHT = 240
WIDTH = 240
DEPTH = 155
N_CHANNELS = 1


class DeepNormalizeDataSet(object):

    def __init__(self, data_dir, modalities, use_fp16=False):
        self.data_dir = data_dir
        self.use_fp16 = use_fp16
        self.modalities = modalities

    def partition_data_set(self, validation_size):
        files = self.fetch_volumes_data_files()
        segs = self.fetch_segmentations()

        return train_test_split(files,
                                segs,
                                test_size=validation_size,
                                shuffle=True,
                                random_state=42)

    def _get_subjects_directories(self):
        return glob.glob(os.path.join(self.data_dir, "*", "*"))

    def fetch_volumes_data_files(self):
        training_data_files = list()

        for subject_dir in self._get_subjects_directories():
            subject_files = list()
            file = subject_dir.split(os.sep)[-1]
            for modality in self.modalities:
                subject_files.append(glob.glob(os.path.join(subject_dir, file + "_" + modality + ".nii.gz"))[0])
            training_data_files.append(subject_files)
        return training_data_files

    def fetch_segmentations(self):
        segmentation_data_files = list()
        for subject_dir in self._get_subjects_directories():
            subject_files = list()
            file = subject_dir.split(os.sep)[-1]
            subject_files.append(glob.glob(os.path.join(subject_dir, file + "_" + "seg" + ".nii.gz"))[0])
            segmentation_data_files.append(subject_files)
        return segmentation_data_files

    def _read_py_function(self, filename, label):

        images, segs, = preprocessing.preprocess_images(filename, label)

        # return nib.load(filename.decode()).get_data().astype(np.float32), nib.load(label.decode()).get_data().astype(np.int32)

        print("Success")
        return images, segs

    def input(self, batch_size, X, y):

        with tf.name_scope("inputs"):

            dataset = tf.data.Dataset.from_tensor_slices((X, y))

            # self._read_py_function(X[0], y[0])

            dataset = dataset.map(
                lambda filename, label: tuple(tf.py_func(
                    self._read_py_function, [filename, label], [np.float32, np.int32])))

            dataset = dataset.repeat()  # Repeat the input indefinitely.

            dataset = dataset.batch(batch_size=batch_size)  # Batch the data set.

            dataset = dataset.prefetch(2 * batch_size)

            return dataset
