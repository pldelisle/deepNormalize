import os
import numpy as np
import nibabel as nib


def get_filename(set_name, case_idx, input_name, loc):
    pattern = '{0}/{1}/{3}/subject-{2}-{3}.nii'
    return pattern.format(loc, set_name, case_idx, input_name)


def read_data(image_path):
    return nib.load(image_path)


def read_vol(input_name):
    image_data = read_data(input_name)
    return image_data.get_data()


def extract_patches(volume, patch_shape, extraction_step, datype='float32'):
    patch_h, patch_w, patch_d = patch_shape[0], patch_shape[1], patch_shape[2]
    stride_h, stride_w, stride_d = extraction_step[0], extraction_step[1], extraction_step[2]
    img_h, img_w, img_d = volume.shape[0], volume.shape[1], volume.shape[2]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_d = (img_d - patch_d) // stride_d + 1
    N_patches_img = N_patches_h * N_patches_w * N_patches_d
    raw_patch_martrix = np.zeros((N_patches_img, patch_h, patch_w, patch_d), dtype=datype)
    k = 0

    # iterator over all the patches
    for h in range((img_h - patch_h) // stride_h + 1):
        for w in range((img_w - patch_w) // stride_w + 1):
            for d in range((img_d - patch_d) // stride_d + 1):
                raw_patch_martrix[k] = volume[h * stride_h:(h * stride_h) + patch_h, \
                                       w * stride_w:(w * stride_w) + patch_w, \
                                       d * stride_d:(d * stride_d) + patch_d]
                k += 1
    assert (k == N_patches_img)
    return raw_patch_martrix


def get_patches_lab(T1_vols, T2_vols, label_vols, extraction_step,
                    patch_shape, validating, testing, num_images_training):
    patch_shape_1d = patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, 2), dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d), dtype="uint8")
    for idx in range(len(T1_vols)):
        y_length = len(y)
        if testing:
            print(("Extracting Patches from Image %2d ....") % (num_images_training + idx + 2))
        elif validating:
            print(("Extracting Patches from Image %2d ....") % (num_images_training + idx + 1))
        else:
            print(("Extracting Patches from Image %2d ....") % (1 + idx))
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step,
                                        datype="uint8")

        # Select only those who are important for processing
        if testing or validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d,
                                    patch_shape_1d, patch_shape_1d, 2), dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d,
                                    patch_shape_1d, patch_shape_1d), dtype="uint8")))

        y[y_length:, :, :, :] = label_patches

        # Sampling strategy: reject samples which labels are only zeros
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step,
                                   datype="float32")
        x[y_length:, :, :, :, 0] = T1_train[valid_idxs]

        # Sampling strategy: reject samples which labels are only zeros
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step
                                   , datype="float32")
        x[y_length:, :, :, :, 1] = T2_train[valid_idxs]
    return x, y


def preprocess_dynamic_lab(dataset,
                           extraction_step,
                           patch_shape,
                           num_images_training=3,
                           validating=True,
                           testing=True,
                           num_images_testing=1):
    if testing:
        print("Testing")
        r1 = num_images_training + 2
        r2 = num_images_training + num_images_testing + 2
        c = num_images_training + 1
        T1_vols = np.empty((num_images_testing, 250, 280, 240), dtype="float32")
        T2_vols = np.empty((num_images_testing, 250, 280, 240), dtype="float32")
        label_vols = np.empty((num_images_testing, 250, 280, 240), dtype="uint8")
    elif validating:
        print("Validating")
        r1 = num_images_training + 1
        r2 = num_images_training + 2
        c = num_images_training
        T1_vols = np.empty((1, 250, 280, 240), dtype="float32")
        T2_vols = np.empty((1, 250, 280, 240), dtype="float32")
        label_vols = np.empty((1, 250, 280, 240), dtype="uint8")
    else:
        print("Training")
        r1 = 1
        r2 = num_images_training + 1
        c = 0
        T1_vols = np.empty((num_images_training, 250, 280, 240), dtype="float32")
        T2_vols = np.empty((num_images_training, 250, 280, 240), dtype="float32")
        label_vols = np.empty((num_images_training, 250, 280, 240), dtype="uint8")

    for case_idx in range(r1, r2):
        print(case_idx)
        T1_vols[(case_idx - c - 1), :, :, :] = read_vol(dataset[0]["t1"][0])
        T2_vols[(case_idx - c - 1), :, :, :] = read_vol(dataset[0]["t2"][0])
        label_vols[(case_idx - c - 1), :, :, :] = read_vol(dataset[0]["label"][0])
    # T1_mean = T1_vols.mean()
    # T1_std = T1_vols.std()
    # T1_vols = (T1_vols - T1_mean) / T1_std
    # T2_mean = T2_vols.mean()
    # T2_std = T2_vols.std()
    # T2_vols = (T2_vols - T2_mean) / T2_std

    # for i in range(T1_vols.shape[0]):
    #     T1_vols[i] = ((T1_vols[i] - np.min(T1_vols[i])) /
    #                   (np.max(T1_vols[i]) - np.min(T1_vols[i]))) * 255
    # for i in range(T2_vols.shape[0]):
    #     T2_vols[i] = ((T2_vols[i] - np.min(T2_vols[i])) /
    #                   (np.max(T2_vols[i]) - np.min(T2_vols[i]))) * 255
    # T1_vols = T1_vols / 127.5 - 1.
    # T2_vols = T2_vols / 127.5 - 1.
    x, y = get_patches_lab(T1_vols, T2_vols, label_vols, extraction_step, patch_shape, validating=validating,
                           testing=testing, num_images_training=num_images_training)
    print("Total Extracted Labelled Patches Shape:", x.shape, y.shape)
    if testing:
        return x, label_vols
    elif validating:
        return x, y, label_vols
    else:
        return x, y


"""
To recompose the image from patches
"""


def recompose3D_overlap(preds, img_h, img_w, img_d, stride_h, stride_w, stride_d):
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    patch_d = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_d = (img_d - patch_d) // stride_d + 1
    N_patches_img = N_patches_h * N_patches_w * N_patches_d
    print("N_patches_h: ", N_patches_h)
    print("N_patches_w: ", N_patches_w)
    print("N_patches_d: ", N_patches_d)
    print("N_patches_img: ", N_patches_img)
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    print("According to the dimension inserted, there are " \
          + str(N_full_imgs) + " full images (of " + str(img_h) + "x" + str(img_w) + "x" + str(img_d) + " each)")
    # itialize to zero mega array with sum of Probabilities
    raw_pred_martrix = np.zeros((N_full_imgs, img_h, img_w, img_d))
    raw_sum = np.zeros((N_full_imgs, img_h, img_w, img_d))
    final_matrix = np.zeros((N_full_imgs, img_h, img_w, img_d), dtype='uint16')

    k = 0
    # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                for d in range((img_d - patch_d) // stride_d + 1):
                    raw_pred_martrix[i, h * stride_h:(h * stride_h) + patch_h, \
                    w * stride_w:(w * stride_w) + patch_w, \
                    d * stride_d:(d * stride_d) + patch_d] += preds[k]
                    raw_sum[i, h * stride_h:(h * stride_h) + patch_h, \
                    w * stride_w:(w * stride_w) + patch_w, \
                    d * stride_d:(d * stride_d) + patch_d] += 1.0
                    k += 1
    assert (k == preds.shape[0])
    # To check for non zero sum matrix
    assert (np.min(raw_sum) >= 1.0)
    final_matrix = np.around(raw_pred_martrix / raw_sum)
    return final_matrix
