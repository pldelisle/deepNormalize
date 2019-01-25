import numpy as np
import nibabel as nib
from DeepNormalize.utils.preprocessing import preprocess_images


if __name__ == '__main__':
    img_data = nib.load(
        "/home/pierre-luc-delisle/Documents/data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_t1.nii.gz").get_data()

    img_data = np.expand_dims(img_data, axis=-1)

    r = preprocess_images(img_data)

    print("hello")