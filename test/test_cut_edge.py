import numpy as np
import nibabel as nib


def cut_edge(data):
    '''Cuts zero edge for a 3D image.

    Args:
        data: A 3D image, [Depth, Height, Width, 1].

    Returns:
        original_shape: [Depth, Height, Width]
        cut_size: A list of six integers [Depth_s, Depth_e, Height_s, Height_e, Width_s, Width_e]
    '''

    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    original_shape = [D, H, W]
    cut_size = [int(D_s), int(D_e + 1), int(H_s), int(H_e + 1), int(W_s), int(W_e + 1)]
    return (original_shape, cut_size)

if __name__ == '__main__':
    img_data = nib.load("/home/pierre-luc-delisle/Documents/data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA08_218_1/Brats18_TCIA08_218_1_t1.nii.gz").get_data()

    r = cut_edge(img_data)

    print("Debug")


