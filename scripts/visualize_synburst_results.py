import os
import sys

# add the based repository path to the system path, to enable absolute imports
repo_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import argparse
from utils.opencv_plotting import BurstSRVis
import torch
import cv2
import numpy as np
from utils.postprocessing_functions import SimplePostProcess
from datasets.synthetic_burst_val_set import SyntheticBurstVal


def visualize_results():
    """ Visualize the results on the SyntheticBurst validation set.
    """

    parser = argparse.ArgumentParser(description='Provides an example on how to use the SyntheticBurst datase')
    parser.add_argument('dataset_path', type=str, help='Path to the Synthetic validation set')
    parser.add_argument('results_path', type=str, help='Path where the results have been saved')
    args = parser.parse_args()

    vis = BurstSRVis(boundary_ignore=40)
    process_fn = SimplePostProcess(return_np=True)

    dataset = SyntheticBurstVal(args.dataset_path)

    for idx in range(len(dataset)):
        burst, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        pred_path = '{}/{}.png'.format(args.results_path, burst_name)
        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        pred = torch.from_numpy(pred.astype(np.float32) / 2 ** 14).permute(2, 0, 1)

        pred = process_fn.process(pred, meta_info)
        data = [{'images': [pred, ],
                 'titles': ['Pred', ]}]
        cmd = vis.plot(data)

        if cmd == 'stop':
            return


if __name__ == '__main__':
    visualize_results()
