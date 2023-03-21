import os
import sys

# add the based repository path to the system path, to enable absolute imports
repo_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import argparse
import torch.nn.functional as F
import cv2
from datasets.synthetic_burst_val_set import SyntheticBurstVal
import torch
import numpy as np


class SimpleBaseline:
    def __init__(self):
        pass

    def __call__(self, burst):
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')
        return burst_rgb


def main():

    parser = argparse.ArgumentParser(description='Provides an example on how to save the results on SyntheticBurstVal dataset for submission on the evaluation server')
    parser.add_argument('in_path', type=str, help='Path to the Synthetic validation set')
    parser.add_argument('out_path', type=str, help='Path where the results are saved')
    parser.add_argument('--device', type=str, default='cuda' ,choices=['cpu','cuda'], help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    dataset = SyntheticBurstVal(args.in_path)

    # TODO Set your network here
    net = SimpleBaseline()


    os.makedirs(args.out_path, exist_ok=True)

    for idx in range(len(dataset)):
        burst, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        burst = burst.to(args.device).unsqueeze(0)

        with torch.no_grad():
            net_pred = net(burst)

        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

        # Save predictions as png
        cv2.imwrite('{}/{}.png'.format(args.out_path, burst_name), net_pred_np)


if __name__ == '__main__':
    main()
