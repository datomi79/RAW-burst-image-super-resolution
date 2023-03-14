# RAW-burst-image-super-resolution
Official repository for the EagleSnap Competition 2023: RAW Burst Image Super Resolution

## Introduction
EagleSnap Super-Resolution Competition is a machine learning challenge aimed at enhancing RAW images. The goal of this competition is to generate a denoised, demosaicked, higher-resolution image, given a RAW burst as input. The competition has two tracks, namelye
- Track 1: Maximize quality based on a simple metric, and
- Track 2: Minimize compute time with quality requirement.

The top ranked participants in each track will be awarded and all participants are invited to submit a report describing their solution.

## Dates

- 01/03/2023 Release of train and validation data
- 01/04/2023 Validation server for Track 1 online
- 01/06/2023 Tool and report submission deadline

## Description

Given multiple noisy RAW images of a scene, the task in EagleSnap Super-Resolution Competition is to predict a denoised higher-resolution RGB image by combining information from the multiple input frames. Concretely, the participants will have a burst sequence containing 14 images to train their model. Each image contains the RAW sensor data from a bayer filter (RGGB) mosaic. The images in the burst have unknown offsets with respect to each other, and are corrupted by noise. The goal is to exploit the information from the multiple input images to predict a denoised, demosaicked RGB image having a 4 times higher resolution, compared to the input


### Track 1: Maximize quality based on a simple metric

In the synthetic track, the input bursts are generated from RGB images using a synthetic data generation pipeline.

#### Evaluation
The proposed methods will be ranked using the fidelity (in terms of PSNR) with the high-resolution ground truth, i.e. the linear sensor space image used to generate the burst. The focus of the challenge is on learning to reconstruct the original high-resolution image, and not the subsequent post-processing. Hence, the PSNR computation will be computed in the linear sensor space, before post-processing steps such as color correction, white-balancing, gamma correction etc.

#### Validation
The results on the validation set can be uploaded on the Codalab server (live now) to obtain the performance measures, as well as a live leaderboard ranking. The results should be uploaded as a ZIP file containing the network predictions for each burst. The predictions must be normalized to the range [0, 2^14] and saved as 16 bit (uint16) png files. Please refer to save_results_synburst_val.py for an example on how to save the results. An example submission file is available here.

#### Final Submission
The test set is now public. You can download the test set containing 92 synthetic bursts from this link. You can use the dataset class provided in synthetic_burst_test_set.py in the latest commit to load the burst sequences.



### Track 2: Minimize compute time with quality requirement

TOOD



## Organizers and Sponsors

ETH Zurich/PBL, Sponsored by Huawei TechArena

- Lukas Cavigelli (lukas.cavigelli@huawei.com)
- Michele Magno (michele.magno@pbl.ee.ethz.ch)
- Luca Pascarella (luca.pascarella@pbl.ee.ethz.ch)
- Davide Plozza (dplozza@student.ethz.ch)

## Terms and conditions
The terms and conditions for participating in the challenge are provided [TODO]([https](https://github.com/dplozza/RAW-burst-image-super-resolution/))

## Acknowledgements
The toolkit uses the forward and inverse camera pipeline code from [unprocessing](https://github.com/timothybrooks/unprocessing).
