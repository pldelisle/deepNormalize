import nibabel as nib
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    img_data = nib.load("/home/pierre-luc-delisle/Documents/data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_t1.nii.gz").get_data()
    img_label = nib.load("/home/pierre-luc-delisle/Documents/data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_seg.nii.gz").get_data()

    