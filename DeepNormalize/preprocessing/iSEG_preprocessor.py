import cupy as cp
import numpy as np
import os
import nibabel as nib

from DeepNormalize.utils.utils import get_iSEG_subjects, save_nifti_image


class iSEGPreprocessor(object):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.modalities_to_preprocess = ["label"]

    @staticmethod
    def _correct_class_ids_iSEG(volume):
        xp = cp.get_array_module(volume)

        # Transfer data from host to device.
        mask = xp.asarray(volume)

        # Keep everything non-background.
        mask[mask == 10] = 1
        mask[mask == 150] = 2
        mask[mask == 250] = 3

        return mask

    def _apply_transformations(self, volume):
        # Load current volume.
        image = nib.load(volume)

        # Load the data.
        image_data = image.get_fdata()

        # Apply required transformations.
        transformed_data = self._correct_class_ids_iSEG(image_data)

        # Enter more transforms here.
        # ...

        return transformed_data, image.affine()

    def preprocess(self):
        iseg_subjects = get_iSEG_subjects(self.input_dir)

        for subject_id, subject in enumerate(iseg_subjects):

            for key in self.modalities_to_preprocess:
                # Get file name.
                filename = os.path.splitext(os.path.basename(subject[key][0]))[0]

                # Apply transforms.
                transformed, affine = self._apply_transformations(subject[key][0])

                # Create new Nifti image.
                new_nifti = nib.Nifti1Image(transformed, affine)
                save_nifti_image(new_nifti,
                                 self.output_dir,
                                 subject_id + 1,
                                 filename + ".nii.gz")
