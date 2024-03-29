import os
import sys

# add the based repository path to the system path, to enable absolute imports
repo_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import argparse
import torch.nn.functional as F
import cv2
from datasets.burstsr_dataset import BurstSRDataset
from torch.utils.data.dataloader import DataLoader
from utils.postprocessing_functions import BurstSRPostProcess
from utils.data_format_utils import convert_dict


def main():

    parser = argparse.ArgumentParser(description='Provides an example on how to use the pre-processed Burst SR dataset')
    parser.add_argument('path', type=str, help='Path to the Burst SR dataset')

    args = parser.parse_args()

    # Load dataset
    dataset = BurstSRDataset(root=args.path,
                             split='val', burst_size=3, crop_sz=56, random_flip=False)

    data_loader = DataLoader(dataset, batch_size=2)

    # Postprocessing function to obtain sRGB images
    postprocess_fn = BurstSRPostProcess(return_np=True)

    for d in data_loader:
        burst, frame_gt, meta_info_burst, meta_info_gt = d

        # A simple baseline which upsamples the base image using bilinear upsampling
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')

        meta_info_gt = convert_dict(meta_info_gt, burst.shape[0])

        # Apply simple post-processing to obtain RGB images
        pred_0 = postprocess_fn.process(burst_rgb[0], meta_info_gt[0])
        gt_0 = postprocess_fn.process(frame_gt[0], meta_info_gt[0])

        pred_0 = cv2.cvtColor(pred_0, cv2.COLOR_RGB2BGR)
        gt_0 = cv2.cvtColor(gt_0, cv2.COLOR_RGB2BGR)

        # Visualize input, ground truth
        cv2.imshow('Input (Demosaicekd + Upsampled)', pred_0)
        cv2.imshow('GT', gt_0)

        input_key = cv2.waitKey(0)
        if input_key == ord('q'):
            return


if __name__ == '__main__':
    main()
