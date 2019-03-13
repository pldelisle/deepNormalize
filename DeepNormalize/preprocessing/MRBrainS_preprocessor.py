import cupy as cp
import numpy as np
import os
import nibabel as nib

from nilearn.image import resample_to_img

from nibabel.processing import resample_to_output

from DeepNormalize.utils.utils import get_MRBrainS_subjects, save_nifti_image


class MRBrainSPreprocessor(object):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.modalities_to_preprocess = ["t1", "t1_ir", "t2"]
        self.labels_keys = ["LabelsForTesting", "LabelsForTraining"]

    # @staticmethod
    # def _pad_image_and_center(volume, x, y, z):
    #     volume = np.asarray(volume)
    #
    #     centered_volume = np.zeros((x, y, z), dtype=np.int32)
    #
    #     d = int(cp.floor(cp.abs(centered_volume.shape[0] - volume.shape[0])) / 2)
    #     e = int(cp.floor(cp.abs(centered_volume.shape[1] - volume.shape[1])) / 2)
    #     f = int(cp.floor(cp.abs(centered_volume.shape[2] - volume.shape[2])) / 2)
    #
    #     centered_volume[d:d + volume.shape[0], e:e + volume.shape[1], f:f + volume.shape[2]] = volume
    #
    #     return centered_volume

    # @staticmethod
    # def _correct_class_ids_MRBrainS(mask):
    #     # Correct labels after interpolation to keep the [0, 3] range.
    #     mask[mask < 0] = 0
    #     mask[mask == 4] = 3
    #
    #     return mask

    @staticmethod
    def _extract_brain(volume, mask):
        """
        Extract brain from head volume.
        This executes on GPU.
        Args:
            volume: Head's volume (Nifti1Image).
            mask: Volume's labels (Nifti1Image).

        Returns:
            Extracted brain.
        """

        data = cp.asarray(volume.get_fdata())
        mask = cp.asarray(mask.get_fdata())

        # Keep everything non-background.
        mask[mask >= 1] = 1

        # Extract brain with segmentation mask.
        brain = cp.multiply(data, mask)

        # Create new Nifti1Image
        extracted_brain = nib.Nifti1Image(brain.get(), volume.affine)

        return extracted_brain

    def _apply_modality_transformations(self, volume_path, label_path, reference_path):
        # Load current volumes.
        volume = nib.load(volume_path)
        label = nib.load(label_path)
        reference = nib.load(reference_path)

        # Apply transforms.
        volume = self._extract_brain(volume, label)
        volume = resample_to_img(volume, reference, interpolation="continuous", clip=False)

        return volume

    def _apply_label_transformations(self, label_path, reference_path):
        # Load current volumes.
        label = nib.load(label_path)
        reference = nib.load(reference_path)

        # Apply transforms.
        volume = resample_to_img(label, reference, interpolation="linear", clip=True)

        return volume

    def preprocess(self):
        mrbrains_subjects = get_MRBrainS_subjects(self.input_dir)

        for subject_id, subject in enumerate(mrbrains_subjects):

            for modality in self.modalities_to_preprocess:
                # Get file name.
                filename = os.path.splitext(os.path.basename(subject[modality][0]))[0]

                # Apply transformations.
                transformed = self._apply_modality_transformations(volume_path=subject[modality][0],
                                                                   label_path=subject["LabelsForTesting"][0],
                                                                   reference_path=subject["t1_1mm"][0])

                save_nifti_image(transformed,
                                 self.output_dir + "/images/" + modality + "/",
                                 subject_id + 1,
                                 filename + ".nii.gz")

            for label in self.labels_keys:
                filename = os.path.splitext(os.path.basename(subject[label][0]))[0]

                # Apply transformations.
                transformed_label = self._apply_label_transformations(label_path=subject[label][0],
                                                                      reference_path=subject["t1_1mm"][0])

                save_nifti_image(transformed_label,
                                 self.output_dir + "/labels/" + label + "/",
                                 subject_id + 1,
                                 filename + ".nii.gz")
