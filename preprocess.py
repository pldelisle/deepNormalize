import argparse
import matplotlib.pyplot as plt

from DeepNormalize.preprocessing.iSEG_preprocessor import iSEGPreprocessor
from DeepNormalize.preprocessing.MRBrainS_preprocessor import MRBrainSPreprocessor


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.show()


def main(args):
    mrbrains = MRBrainSPreprocessor(args.mrbrains_data_dir, args.mrbrains_output_dir)
    mrbrains.preprocess()

    iseg = iSEGPreprocessor(args.iseg_data_dir, args.iseg_output_dir)
    iseg.preprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iseg-data-dir",
        type=str,
        required=True,
        help="The directory where the iSEG input data is stored.")
    parser.add_argument(
        "--mrbrains-data-dir",
        type=str,
        required=True,
        help="The directory where the MRBRainS input data is stored.")
    parser.add_argument(
        "--iseg-output-dir",
        type=str,
        required=True,
        help="The directory where the output data is stored.")
    parser.add_argument(
        "--mrbrains-output-dir",
        type=str,
        required=True,
        help="The directory where the output data is stored.")
    args = parser.parse_args()

    main(args=args)
