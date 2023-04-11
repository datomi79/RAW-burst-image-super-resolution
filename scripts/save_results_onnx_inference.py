import os
import sys

# add the based repository path to the system path, to enable absolute imports
repo_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import argparse
import time
import cv2
import numpy as np
import torch
import onnxruntime

from datasets.synthetic_burst_test_set import SyntheticBurstTest



def main():
    """ Example usage: python scripts/save_results_onnx_inference.py ./data/test_set ./data/test_output ./data/simple_baseline.onnx
    """ 

    parser = argparse.ArgumentParser(description='Shows how the onnx models will be used for inference for Track 2 evaluation')
    parser.add_argument('in_path', type=str, help='Path to the test set for Track 2')
    parser.add_argument('out_path', type=str, help='Path where the results are saved')
    parser.add_argument('onnx_model_path', type=str, help='Filename of the onnx model')
    args = parser.parse_args()


    # Load the test set
    dataset = SyntheticBurstTest(args.in_path)


    # Create output folder if it does not exist
    os.makedirs(args.out_path, exist_ok=True)


    N = len(dataset)

    # create an instance of the ONNX runtime inference session
    ort_session = onnxruntime.InferenceSession(args.onnx_model_path)

    # make one warm-up inference
    burst, meta_info = dataset[0]
    burst = burst.unsqueeze(0)

    ort_inputs = {ort_session.get_inputs()[0].name: burst.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    total_time = 0.0
    for idx in range(N):
        burst, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']
        print(burst_name)

        burst = burst.unsqueeze(0)
        
        # run and time inference using the ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: burst.numpy()}

        start_time = time.perf_counter()
        ort_outputs = ort_session.run(None, ort_inputs)
        end_time = time.perf_counter()

        # calculate the elapsed time
        elapsed_time = end_time - start_time
        print(elapsed_time)
        total_time += elapsed_time

        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (torch.tensor(ort_outputs[0]).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

        # Save predictions as png
        cv2.imwrite('{}/{}.png'.format(args.out_path, burst_name), net_pred_np)
        
        

    # calculate the average inference time
    avg_time = total_time / N
    print("Average inference time: {:.4f} seconds".format(avg_time))


if __name__ == '__main__':
    main()
