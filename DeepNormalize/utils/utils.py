import os
import re
import random
import nibabel as nib
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def get_MRBrainS_data(path):
    """
    Utility method to get, filter and arrange BraTS data set in a series of lists.
    Args:
        path: The path to look for BraTS data set files.

    Returns:
        A tuple containing multimodal MRI images for each subject and their respective segmentation.
    """
    subjects = list()
    keys = ["t1", "t1_1mm", "t1_ir", "t2", "roi", "LabelsForTraining", "label"]

    for (dirpath, dirnames, filenames) in os.walk(path):
        if len(filenames) is not 0:
            # Filter files.
            t1 = list(filter(re.compile(r"^T1.nii").search, filenames))
            t1_1mm = list(filter(re.compile(r"^T1_1mm.nii").search, filenames))
            t1_ir = list(filter(re.compile(r"^T1_IR.nii").search, filenames))
            t2 = list(filter(re.compile(r"^T2_FLAIR.nii").search, filenames))
            roi = list(filter(re.compile(r"^ROIT1.nii").search, filenames))
            seg_training = list(filter(re.compile(r"^LabelsForTraining.nii").search, filenames))
            seg_testing = list(filter(re.compile(r"^LabelsForTesting.nii").search, filenames))

            t1 = [os.path.join(dirpath, ("{}".format(i))) for i in t1]
            t1_1mm = [os.path.join(dirpath, ("{}".format(i))) for i in t1_1mm]
            t1_ir = [os.path.join(dirpath, ("{}".format(i))) for i in t1_ir]
            t2 = [os.path.join(dirpath, ("{}".format(i))) for i in t2]
            roi = [os.path.join(dirpath, ("{}".format(i))) for i in roi]
            seg_training = [os.path.join(dirpath, ("{}".format(i))) for i in seg_training]
            seg_testing = [os.path.join(dirpath, ("{}".format(i))) for i in seg_testing]

            subjects.append(dict((key, volume) for key, volume in zip(keys, [t1,
                                                                             t1_1mm,
                                                                             t1_ir,
                                                                             t2,
                                                                             roi,
                                                                             seg_training,
                                                                             seg_testing])))

    return subjects


def get_iSEG_data(path):
    """
        Utility method to get, filter and arrange BraTS data set in a series of lists.
        Args:
            path: The path to look for BraTS data set files.

        Returns:
            A tuple containing multimodal MRI images for each subject and their respective segmentation.
        """

    subjects = list()
    keys = ["t1", "t2", "roi", "label"]

    for (dirpath, dirnames, filenames) in os.walk(path):
        if len(filenames) is not 0:
            # Filter files.
            t1 = list(filter(re.compile(r"^.*?T1.nii$").search, filenames))
            t2 = list(filter(re.compile(r"^.*?T2.nii$").search, filenames))
            roi = list(filter(re.compile(r"^.*?ROIT1.nii.gz$").search, filenames))
            seg_training = list(filter(re.compile(r"^.*?label.nii$").search, filenames))

            t1 = [os.path.join(dirpath, ("{}".format(i))) for i in t1]
            t2 = [os.path.join(dirpath, ("{}".format(i))) for i in t2]
            roi = [os.path.join(dirpath, ("{}".format(i))) for i in roi]
            seg_training = [os.path.join(dirpath, ("{}".format(i))) for i in seg_training]

            subjects.append(dict((key, volume) for key, volume in zip(keys, [t1,
                                                                             t2,
                                                                             roi,
                                                                             seg_training])))

    return subjects


def save_nifti_image(new_image, output_dir, subject_id, filename):
    """
    Save the preprocessed image.
    Args:
        subject_id: Will save in the subject_id folder.
        filename: The new file name.
        new_image: A Nifti1Image object containing header and data.

    Returns:
        None

    """
    if not os.path.exists(os.path.join(output_dir, str(subject_id))):
        os.makedirs(os.path.join(output_dir, str(subject_id)))
    filepath = os.path.join(os.path.join(os.path.join(output_dir, str(subject_id))), filename)
    nib.save(new_image, filepath)


def train_test_split(dataset):
    random.shuffle(dataset)

    X_train = dataset[:-2]
    X_valid = [dataset[-2]]
    X_test = [dataset[-1]]

    return X_train, X_valid, X_test


def correct_class_ids(segmenation_volume):
    # Transfer data from host to device.
    mask_gpu = cp.asarray(segmenation_volume)

    # Keep everything non-background.
    mask_gpu[mask_gpu == 10] = 1
    mask_gpu[mask_gpu == 150] = 2
    mask_gpu[mask_gpu == 250] = 3

    return mask_gpu.get()


def generate_ROI(volume):
    """
    Returns a region of interest (ROI) map where voxel intensity is higher than 0.
    :param volume: A 3D image.
    :return: A 3D array containing the region of interest (1 if condition is verified).
    """

    volume_gpu = cp.asarray(volume)

    idx = cp.where(volume_gpu > 0)

    roiVolume = cp.zeros(volume_gpu.shape, dtype=cp.float32)

    roiVolume[idx] = 1

    return roiVolume.get()


def export_inputs(data):
    for i, subject in enumerate(range(data[0].shape[0])):

        if not os.path.exists(os.path.join("outputs", str(i))):
            os.makedirs(os.path.join("outputs", str(i)))

        for modality in range(data[0].shape[1]):
            patch = data[0][i][modality].astype(np.int32)
            plt.imshow(patch[:, :, 32], cmap='gray')
            plt.savefig("outputs/" + str(i) + "/T" + str(modality + 1) + ".png")

        seg = np.squeeze(data[2][i]).astype(np.int32)
        plt.imshow(seg[:, :, 32], cmap='gray')
        plt.savefig("outputs/" + str(i) + "/seg.png")
