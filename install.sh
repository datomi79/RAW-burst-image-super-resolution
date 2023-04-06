#!/bin/bash

echo "****************** Installing pytorch ******************"
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch


echo "****************** Installing onnx runtime ******************"
conda install -c conda-forge onnxruntime onnx
pip install onnx

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** exifread ******************"
pip install exifread


echo ""
echo ""
echo "****************** Installation complete! ******************"
