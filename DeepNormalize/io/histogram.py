import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def get_histogram(image, filename):
    image = nib.load(image).get_data().astype(np.float32)

    result = image.flatten()


    n, bins, patches = plt.hist(result, bins=1024, range=(result.min(), result.max()), density=True, facecolor='red',
                                alpha=0.75,
                                histtype='step')

    plt.savefig(filename)


if __name__ == '__main__':
    get_histogram(
        "/Users/pierre-luc-delisle/Documents/research/test_shit_5.nii.gz", "release_shit_5.png")
    # get_histogram(
    #     "/Users/pierre-luc-delisle/Documents/Data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA04_192_1/Brats18_TCIA04_192_1_t1.nii.gz",
    #     "official.png")
    # get_histogram("/Users/pierre-luc-delisle/Documents/Data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA01_412_1/Brats18_TCIA01_412_1_t1.nii.gz")
    #
    # get_histogram(
    #     "/Users/pierre-luc-delisle/Documents/Data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_208_1/Brats18_TCIA02_208_1_t1.nii.gz")
    #
