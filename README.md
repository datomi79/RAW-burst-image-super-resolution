# EagleSnap Competition 2023: RAW Burst Image Super Resolution
Official repository for the EagleSnap Competition 2023: RAW Burst Image Super Resolution

## Introduction
EagleSnap Super-Resolution Competition is a machine learning challenge aimed at enhancing RAW images. The goal of this competition is to generate a denoised, demosaicked, higher-resolution image, given a RAW burst as input. The competition has two tracks, namely
- Track 1: Maximize quality based on a simple metric.

The top ranked participants in each track will be awarded and all participants are invited to submit a report describing their solution.

## Dates

- 01/03/2023 Release of train and validation data
- 01/04/2023 Validation server for Track 1 online
- 01/05/2023 Release of test data for Track 1
- 01/06/2023 Tool and report submission deadline

## Description

Given multiple noisy RAW images of a scene, the task in EagleSnap Super-Resolution Competition is to predict a denoised higher-resolution RGB image by combining information from the multiple input frames. Concretely, the participants will have a burst sequence containing 14 images to train their model. Each image contains the RAW sensor data from a bayer filter (RGGB) mosaic. The images in the burst have unknown offsets with respect to each other, and are corrupted by noise. The goal is to exploit the information from the multiple input images to predict a denoised, demosaicked RGB image having a 4 times higher resolution, compared to the input


### Track 1: Maximize quality based on a simple metric

In the synthetic track, the input bursts are generated from RGB images using a synthetic data generation pipeline.

**Data generation:** The input sRGB image is first converted to linear sensor space 
using an inverse camera pipeline. A LR burst is then generated by applying random 
translations and rotations, followed by bilinear downsampling. The generated burst is 
then mosaicked and corrupted by random noise. 

**Training set:** We provide [code](datasets/synthetic_burst_train_set.py) to generate the synthetic 
bursts using any image dataset for training. Note that any image dataset **except the 
validation split of the [BurstSR dataset](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf)** can be used to generate synthetic bursts for training.  

**Validation set:** The bursts in the validation set have been 
pre-generated with the [data generation code](datasets/synthetic_burst_train_set.py), 
using the RGB images from the validation split of the 
raw BurstSR dataset. The dataset can be downloaded from [here](https://figshare.com/articles/dataset/synburst_val_2023_zip/22439116).

#### Registration
If you wish to participate in the Synthetic track, please register for the challenge at the 
[codalab page](https://codalab.lisn.upsaclay.fr/competitions/12007#learn_the_details) to get access to the evaluation server and receive email notifications for the challenge.

#### Evaluation
The proposed methods will be ranked using the fidelity (in terms of PSNR) with the high-resolution ground truth, i.e. the linear sensor space image used to generate the burst. The focus of the challenge is on learning to reconstruct the original high-resolution image, and not the subsequent post-processing. Hence, the PSNR computation will be computed in the linear sensor space, before post-processing steps such as color correction, white-balancing, gamma correction etc.

#### Validation
The results on the validation set can be uploaded on the [Codalab server](https://codalab.lisn.upsaclay.fr/competitions/12007#learn_the_details) to obtain the performance measures, as well as a live leaderboard ranking. The results should be uploaded as a ZIP file containing the network predictions for each burst. The predictions must be normalized to the range [0, 2^14] and saved as 16 bit (uint16) png files. Please refer to [save_results_synburst_val.py](scripts/save_results_synburst_val.py) for an example on how to save the results. An example submission file is available [here](https://figshare.com/articles/dataset/synburst_example_submission_2023_zip/22439179).

#### Final Submission
<!-- The test set is now public. You can download the test set containing 92 synthetic bursts from this [link][TODO]. -->
The test set will be made public in a later date, see [Dates](#dates).
You can use the dataset class provided in [synthetic_burst_test_set.py](datasets/synthetic_burst_test_set.py) in the latest commit to load the burst sequences.

For the final submission, you need to submit:
* The predicted outputs for each burst sequence as a zip folder, in the same format as used for uploading results to the codalab validation server.
* The code and model files necessary to reproduce your results.
* A factsheet (both PDF and tex files) describing your method. The template for the factsheet is available [here](https://figshare.com/articles/media/RAW_Burst_Image_Super_Resolution_Template_zip/22439071).  

The results, code, and factsheet should be submitted via a google form, which will be made available with the release of test data for Track 1.
<!-- The results, code, and factsheet should be submitted via the [google form] [TODO](https://docs.google.com/forms/d/e/1FAIpQLSduZNcb6M-e_ROEnATRJ7e58ChUrLgrQ7iSmS6ysoON3wHZqg/viewform?usp=sf_link) -->

## Toolkit
We also provide a Python toolkit which includes the necessary data loading and 
evaluation scripts. The toolkit contains the following modules.

* [data_processing](data_processing): Contains the forward and inverse camera pipeline 
  employed in [“Unprocessing images for learned raw denoising”](https://arxiv.org/abs/1811.11127), 
  as well as the [script](data_processing/synthetic_burst_generation.py) to generate a 
  synthetic burst from a single RGB image.
* [datasets](datasets): Contains the PyTorch dataset classes useful for the challenge. 
    * [synthetic_burst_train_set](datasets/synthetic_burst_train_set.py) provides the SyntheticBurst dataset which generates synthetic bursts using any image dataset. 
    * [zurich_raw2rgb_dataset](datasets/zurich_raw2rgb_dataset.py) can be used to load 
      the RGB images Zurich RAW to RGB mapping dataset. This can be used along with SyntheticBurst dataset to generate synthetic bursts for training.  	
    * [synthetic_burst_val_set](datasets/synthetic_burst_val_set.py) can be used to load 
      the pre-generated synthetic validation set.
    * [synthetic_burst_test_set](datasets/synthetic_burst_test_set.py) can be used to load 
      the pre-generated synthetic test set.
    <!---* [realworld_burst_test_set](datasets/realworld_burst_test_set.py) can be used to load 
      the real world bursts for track 2 test set.--->
    * [burstsr_dataset](datasets/burstsr_dataset.py) provides the BurstSRDataset class which can be used to load the RAW bursts and high-resolution ground truths
   from the pre-processed BurstSR dataset.
* [scripts](scripts): Includes useful example scripts.
    * [download_burstsr_dataset](scripts/download_burstsr_dataset.py) can be used to 
      download and unpack the BurstSR dataset.
    * [download_raw_burstsr_data](scripts/download_raw_burstsr_data.py) can be used to download the unprocessed BurstSR dataset.
    * [test_synthetic_burst](scripts/test_synthetic_bursts.py) provides an example on how
  to use the [SyntheticBurst](datasets/synthetic_burst_train_set.py) dataset.
    * [test_burstsr_dataset](scripts/test_burstsr_dataset.py) provides an example on how
  to use the pre-processed [BurstSR](datasets/burstsr_dataset.py) dataset.
    * [save_results_synburst_val](scripts/save_results_synburst_val.py) provides an example
      on how to save the results on [SyntheticBurstVal](datasets/synthetic_burst_val_set.py) 
      dataset for submission on the evaluation server.
    * [save_results_synburst_test](scripts/save_results_synburst_test.py) provides an example
      on how to save the results on [SyntheticBurstTest](datasets/synthetic_burst_test_set.py) 
      dataset for the final submission.
    <!---* [save_results_realworld_test](scripts/save_results_realworld_test.py) provides an example
      on how to save the results on [RealWorldBurstTest](datasets/realworld_burst_test_set.py) 
      dataset for the final submission.--->
    * [visualize_synburst_results](scripts/visualize_synburst_results.py) Visualize generated results on the synthetic burst 
  validation set.
    
* [utils](utils): Contains utility functions.

**Installation:** The toolkit requires [PyTorch](https://pytorch.org/) and [OpenCV](https://opencv.org/) 
for track 1. The necessary packages can be installed with [anaconda](https://www.anaconda.com/), using the [install.sh](install.sh) script. 


## Data
We provide the following data as part of the challenge. 

**Synthetic validation set:** The official validation set for track 1. The dataset contains 100 synthetic bursts, each containing 
14 RAW images of 256x256 resolution. The synthetic bursts are generated from the RGB Canon images from the validation split of the BurstSR dataset. 
The dataset can be downloaded from [here](https://figshare.com/articles/dataset/synburst_val_2023_zip/22439116).

**Synthetic test set:** The official test set for track 1. The dataset contains 92 synthetic bursts, each containing 
14 RAW images of 256x256 resolution. The synthetic bursts are generated from the RGB Canon images from the test split of the BurstSR dataset.
The test set will be made avaliable at a later date, see [Dates](#dates).
<!-- The dataset can be downloaded from [here][TODO](https://data.vision.ee.ethz.ch/bhatg/synburst_test_2022.zip). -->

**BurstSR train and validation set (pre-processed):** The dataset has been split into 10 parts and can be downloaded and unpacked using the 
[download_burstsr_dataset.py](scripts/download_burstsr_dataset.py) script. In case of issues with the script, the download links 
are available [here](data_specs/burstsr_links.md).

**BurstSR train and validation set (raw):** The dataset can be downloaded and unpacked using the [scripts/download_raw_burstsr_data.py](scripts/download_raw_burstsr_data.py) script. In case of issues with the script, the download links 
are available [here](data_specs/burstsr_links.md).

**Zurich RAW to RGB mapping set:** The RGB images from the training split of the 
[Zurich RAW to RGB mapping dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset) 
can be downloaded from [here](https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip). These RGB images can be 
used to generate synthetic bursts for training using  the SyntheticBurst class.


## Organizers and Sponsors

ETH Zurich/PBL, Sponsored by Huawei TechArena

- Lukas Cavigelli (lukas.cavigelli@huawei.com)
- Michele Magno (michele.magno@pbl.ee.ethz.ch)
- Luca Pascarella (luca.pascarella@pbl.ee.ethz.ch)
- Davide Plozza (davide.plozza@pbl.ee.ethz.ch)

## Terms and conditions
The terms and conditions for participating in the challenge are provided here [TODO].

## Acknowledgements
The toolkit was largely adapted from the [NTIRE22_BURSTSR](https://github.com/goutamgmb/NTIRE22_BURSTSR) challenge repository.

The toolkit uses the forward and inverse camera pipeline code from [unprocessing](https://github.com/timothybrooks/unprocessing).
