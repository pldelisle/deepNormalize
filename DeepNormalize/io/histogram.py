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
For plotting image histogram purpose.

"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def get_histogram(volume_file_path, output_filename):
    """
    Get histogram of an image and save it on disk.
    :param volume_file_path: A Nifti volume file path.
    :param output_filename: An ouput file name to save the histogram figure.
    :return: None.
    """
    volume_file_path = nib.load(volume_file_path).get_data().astype(np.float32)

    result = volume_file_path.flatten()


    n, bins, patches = plt.hist(result, bins=1024, range=(result.min(), result.max()), density=True, facecolor='red',
                                alpha=0.75,
                                histtype='step')

    plt.savefig(output_filename)


if __name__ == '__main__':
    get_histogram(
        "/Users/pierre-luc-delisle/Documents/Data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA04_192_1/Brats18_TCIA04_192_1_t1.nii.gz",
        "official.png")
    get_histogram("/Users/pierre-luc-delisle/Documents/Data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA01_412_1/Brats18_TCIA01_412_1_t1.nii.gz")

    get_histogram(
        "/Users/pierre-luc-delisle/Documents/Data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_208_1/Brats18_TCIA02_208_1_t1.nii.gz")

