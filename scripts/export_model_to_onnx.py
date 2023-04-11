import os
import sys

# add the based repository path to the system path, to enable absolute imports
repo_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import argparse
import torch.nn.functional as F
import cv2
import torch
import numpy as np
import onnx
import onnxruntime



class SimpleBaseline(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, burst):
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')
        return burst_rgb


def main():
    """ Example usage: python scripts/export_model_to_onnx.py ./data/simple_baseline.onnx
    """ 

    parser = argparse.ArgumentParser(description='Provides an example on how to convert and export a model in a valid onnx format')
    parser.add_argument('onnx_model_path', type=str, help='Filename of the output onnx model')
    args = parser.parse_args()

    device = "cpu"


    # Create output folder if it does not exist
    os.makedirs(os.path.dirname(args.onnx_model_path), exist_ok=True)


    # TODO Set your network here
    model = SimpleBaseline()

    # create an example input tensor
    input_shape = (1, 14, 4, 128, 128)  # shape of a burst input
    input_tensor = torch.randn(input_shape)


    # export the model to ONNX
    torch.onnx.export(
        model, 
        input_tensor, 
        args.onnx_model_path,
        opset_version=12,
    )

    # check that the exported model is valid
    onnx_model = onnx.load(args.onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # create an instance of the ONNX runtime inference session
    ort_session = onnxruntime.InferenceSession(args.onnx_model_path)

    # run inference using the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # compare the PyTorch and ONNX model outputs
    with torch.no_grad():
        torch_output = model(input_tensor.to(device)).numpy()
        
        # make sure that the original model and the onnx version have the same output
        np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-03, atol=1e-05)

    # make sure that the output has the right shape
    assert(ort_outputs[0].shape ==  (1, 3, 1024, 1024))


    print("Exported model has been validated, and has the correct output shape.")


if __name__ == '__main__':
    main()
