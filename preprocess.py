import cupy as cp
import os
import nibabel as nib
import argparse
import re

from nibabel.processing import resample_to_output

from DeepNormalize.utils.utils import get_MRBrainS_data, save_nifti_image


def extract_brain(volume, mask):
    """
    Extract brain from head volume.
    This executes on GPU.
    Args:
        volume: Head's volume (Nifti1Image).
        mask: Volume's labels (Nifti1Image.

    Returns:
        Extracted brain.
    """

    # Transfer data from host to device.
    image_gpu = cp.asarray(volume.get_data())
    mask_gpu = cp.asarray(mask.get_data())

    # Keep everything non-background.
    mask_gpu[mask_gpu >= 1] = 1

    # Extract brain with segmentation mask.
    brain = cp.multiply(image_gpu, mask_gpu)

    # Transfer data from device to host.
    extracted_brain = brain.get()

    # Create new Nifti1Image
    extracted_brain = nib.Nifti1Image(extracted_brain, volume.affine)

    return extracted_brain


def main(args):
    subjects = get_MRBrainS_data(args.data_dir)

    keys = ["t1", "t1_ir", "t2", "roi", "LabelsForTraining", "label"]

    for subject_id, subject in enumerate(subjects):

        label = nib.load(subject["label"][0])

        for key in keys:
            # Get file name.
            filename = os.path.basename(subject[key][0])

            # Load current volume.
            image = nib.load(subject[key][0])

            # Apply transformations.
            if key != "label" or "LabelsForTraining":
                image = extract_brain(image, label)

            image = resample_to_output(image, order=3)

            save_nifti_image(image,
                             args.output_dir,
                             subject_id + 1,
                             filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="The directory where the MRBrainS input data is stored.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory where the output data is stored.")
    args = parser.parse_args()

    main(args=args)
