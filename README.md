# EagleSnap Competition 2023: RAW Burst Image Super Resolution
Official repository for the EagleSnap Competition 2023: RAW Burst Image Super Resolution

## Introduction
EagleSnap Super-Resolution Competition is a machine learning challenge aimed at enhancing RAW images. The goal of this competition is to generate a denoised, demosaicked, higher-resolution image, given a RAW burst as input. The competition has two tracks, namely
- Track 1: Maximize quality based on a simple metric.
- Track 2: Minimize inference time while maximizing quality.

The top ranked participants in each track will be awarded and all participants are invited to submit a report describing their solution.

## Dates

- 29/05/2023: Start of Track 1.
- 12/06/2023: Start of Track 2.
- 31/07/2023: Track 1 & 2 submission deadline.
- End of August: Notification of winners.
- 20/09/2023: Award ceremony.

## Prizes

The prizes will be awarded in the form of vouchers (e.g., Amazon vouchers) with the following values:

| Track | 1<sup>st</sup> Prize | 2<sup>nd</sup> Prize | 3<sup>rd</sup> Prize |
|-------|-------------|-------------|-------------|
| 1     | CHF 1500    | CHF 800     | CHF 400     |
| 2     | CHF 1500    | CHF 800     | CHF 400     | 

## Description

Given multiple noisy RAW images of a scene, the task in EagleSnap Super-Resolution Competition is to predict a denoised higher-resolution RGB image by combining information from the multiple input frames. Concretely, the participants will have a burst sequence containing 14 images to train their model. Each image contains the RAW sensor data from a bayer filter (RGGB) mosaic. The images in the burst have unknown offsets with respect to each other, and are corrupted by noise. The goal is to exploit the information from the multiple input images to predict a denoised, demosaicked RGB image having a 4 times higher resolution, compared to the input


### Track 1: Maximize quality based on a simple metric

In this track, the input bursts are generated from RGB images using a synthetic data generation pipeline. The goal of this track is to focus on improving image quality, without any inference speed constraints. Participants are required to propose models that can generate high-quality super-resolved images from the given burst sequences.

**Data generation:** The input sRGB image is first converted to linear sensor space 
using an inverse camera pipeline. A LR burst is then generated by applying random 
translations and rotations, followed by bilinear downsampling. The generated burst is 
then mosaicked and corrupted by random noise. 

**Training set:** We provide [code](datasets/synthetic_burst_train_set.py) to generate the synthetic 
bursts using any image dataset for training. Note that any public image dataset **except the 
validation split of the [BurstSR dataset](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf)** can be used to generate synthetic bursts for training. The dataset has to be publicly accessible. Unpublished or self-made datasets are not allowed.

**Validation set:** The bursts in the validation set have been 
pre-generated with the [data generation code](datasets/synthetic_burst_train_set.py), 
using the RGB images from the validation split of the 
raw BurstSR dataset. The dataset can be downloaded from [here](https://figshare.com/articles/dataset/synburst_val_2023_zip/22439116).

**Test set:** The bursts in the validation set have been generated with the [data generation code](datasets/synthetic_burst_train_set.py), from unpublished RGB images taken with a Canon DSLR camera. [Data](#data) for more details. 

#### Registration

Refer to the [Competition Rules](#competition-rules).

#### Evaluation
The proposed methods will be ranked using the fidelity (in terms of PSNR) with the high-resolution ground truth, i.e. the linear sensor space image used to generate the burst. The focus of the challenge is on learning to reconstruct the original high-resolution image, and not the subsequent post-processing. Hence, the PSNR computation will be computed in the linear sensor space, before post-processing steps such as color correction, white-balancing, gamma correction etc.

#### Validation
The results on the validation set can be uploaded on the Codalab server, which will be made avaliable soon, <!-- [Codalab server](https://codalab.lisn.upsaclay.fr/competitions/12007#learn_the_details) --> to obtain the performance measures, as well as a live leaderboard ranking. The results should be uploaded as a ZIP file containing the network predictions for each burst. The predictions must be normalized to the range [0, 2^14] and saved as 16 bit (uint16) png files. Please refer to [save_results_synburst_val.py](scripts/save_results_synburst_val.py) for an example on how to save the results. An example submission file is available [here](https://figshare.com/articles/dataset/synburst_example_submission_2023_zip/22439179).

#### Final Submission
<!-- The test set is now public. You can download the test set containing 92 synthetic bursts from this [link][TODO]. -->
The test set will be made publib shortly.
You can use the dataset class provided in [synthetic_burst_test_set.py](datasets/synthetic_burst_test_set.py) to load the burst sequences.

For the final submission, you need to submit:
* The predicted outputs for each burst sequence as a zip folder, in the same format as used for uploading results to the codalab validation server.
* The code and model files necessary to reproduce your results. You should provide clear instructions on how to download the datasets and any external dependencies. The organizers need to have all the necessary resources in order to replicate both training and inference.
* A factsheet (both PDF and tex files) describing your method. The template for the factsheet is available [here](https://figshare.com/articles/media/RAW_Burst_Image_Super_Resolution_Template_zip/22439071).  

The results, code, and factsheet should be submitted via a Google Form, which will be made available with the release of test data for Track 1.
<!-- The results, code, and factsheet should be submitted via the [Google Form] [TODO](https://docs.google.com/forms/d/e/1FAIpQLSduZNcb6M-e_ROEnATRJ7e58ChUrLgrQ7iSmS6ysoON3wHZqg/viewform?usp=sf_link) -->

### Track 2: Minimize inference time while maximizing quality

The goal of this track is to optimize the trade-off between image quality and inference speed, which is particularly important for real-time applications. To this end, participants are required to propose models that can generate high-quality super-resolved images with fast inference time. 

**Training set:** The training set and data generation are the same as in [Track 1](#track-1-maximize-quality-based-on-a-simple-metric).


#### Registration

Refer to the [Competition Rules](#competition-rules).

<!--If you wish to participate in Track 2, please register for the challenge at the **TODO** [codalab page](test) to get access to the evaluation server and receive email notifications for the challenge.-->

#### Evaluation
The models will be evaluated based on a combination of PSNR and inference speed. Specifically, we will use the following metric to rank the models:

$$
score = PSNR - 4*log_{10}(T_{inference})
$$

The PSNR will be computed on the test set similarly to [Track 1](#track-1-maximize-quality-based-on-a-simple-metric), while $T_{inference}$ is the average inference time of the model computed on the test set.
The higher the score, the better the model. This metric aims to balance the trade-off between image quality and inference speed. 


#### Final Submission

For this track there will be no online validation server.
The test set will be made public at a later date, see [Dates](#dates). 

The score will be computed using the [ONNX runtime](https://onnxruntime.ai/) on the CPU. The partecipants will be required to provide an ONNX model that can be run with the [save_results_onnx_inference.py](scripts/save_results_onnx_inference.py). This scripts also shows how the test set prediction will be generated for scoring (same format as [Track 1](#track-1-maximize-quality-based-on-a-simple-metric)), as well as an inference time measurement example.
The final execution time will be computed with a more sophisticated procedure on a workstation with an Intel i9 13900k CPU. 

For the final submission, participants need to submit:
* An ONNX model that can be run with [save_results_onnx_inference.py](scripts/save_results_onnx_inference.py).
* The code and model files necessary to reproduce their results.
* A factsheet (both PDF and tex files) describing their method. The template for the factsheet is available [here](https://figshare.com/articles/media/RAW_Burst_Image_Super_Resolution_Template_zip/22439071).

The results, code, and factsheet should be submitted via a google form, which will be made available with the release of test data for Track 2.


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
      the pre-generated synthetic test set, for both Track 1 and 2.
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
      dataset for the final Track 1 submission.
    <!---* [save_results_realworld_test](scripts/save_results_realworld_test.py) provides an example
      on how to save the results on [RealWorldBurstTest](datasets/realworld_burst_test_set.py) 
      dataset for the final submission.--->
    * [visualize_synburst_results](scripts/visualize_synburst_results.py) Visualize generated results on the synthetic burst validation set.
    * [export_model_to_onnx](scripts/export_model_to_onnx.py) Example on how to export a PyTorch model to ONNX format (for Track 2).
    * [save_results_onnx_inference.py](scripts/save_results_onnx_inference.py) Shows how the onnx models will be used for inference for Track 2 evaluation.
  
    
* [utils](utils): Contains utility functions.

**Installation:** The toolkit requires [PyTorch](https://pytorch.org/) and [OpenCV](https://opencv.org/) for Track 1 & 2, as well as the [ONNX](https://onnxruntime.ai/) runtime for Track 2. The necessary packages can be installed with [anaconda](https://www.anaconda.com/), using the [install.sh](install.sh) script. 


## Data
We provide the following data as part of the challenge. 

**Track 1 validation set:** The official validation set for Track 1. The dataset contains 100 synthetic bursts, each containing 
14 RAW images of 1024x1024 resolution. The synthetic bursts are generated from the RGB Canon images from the validation split of the BurstSR dataset. 
The dataset can be downloaded from [here](https://figshare.com/articles/dataset/synburst_val_2023_zip/22439116).

**Track 1 & 2 test set:** The official test set for Track 1 & 2. The dataset contains 80 synthetic bursts, each containing 
14 RAW images of 1024x1024 resolution. The test set is generated from unpublished RGB images taken with a Canon DSLR camera.
The test set will be made avaliable soon.
<!-- The dataset can be downloaded from [here](https://figshare.com/articles/dataset/synburst_val_2023_zip/22439116). -->

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

## Competition Rules

Please read these terms and conditions carefully before participating in the coding competition.
By registering for the competition, you agree to be bound by the following terms and conditions:

1. **Submission by Individuals or Teams:** A submission can be made by either a single person or a team consisting of an unlimited number of individuals. In the case of a team submission, the prize awarded will be shared among the team members. It is the responsibility of the team members to mutually decide on the distribution of any prizes received. The competition organizers will not be involved in the internal allocation of prizes within a team.
2. **Eligibility:** At least one member of the team must be a registered student at a Swiss university during the sign-up process for the competition.
3. **Team Limitation:** Each person can participate in a maximum of one team. Multiple registrations or participation in multiple teams will result in disqualification.
4. **Team Composition and Code Sharing:** All contributors to a submission are considered team members. If a team utilizes code, programs, or libraries that are not self-authored, the source code or binary of such external resources must be publicly accessible. This ensures that other participating teams have access to the same resources and can review the code used in the submission.
5. **Public Datasets Usage:** Participants are allowed to use only public datasets that are accessible to anyone. The source and details of the datasets used must be clearly disclosed in the submission.
6. **Prize Eligibility:** All team members must be eligible to receive the prizes as per the laws and regulations of their respective countries. Participants must ensure that their involvement in the competition does not violate any government restrictions or sanctions. Failure to comply with this requirement may result in disqualification and forfeiture of any prizes awarded.
7. **Multiple Tracks:** The competition consists of two tracks, namely Track A and Track B. Teams have the option to participate in either one or both tracks. However, registration must be made individually for each track. Teams cannot submit the same project for both tracks.
8. **Registration and Submission Process:** Official registration for each track of the coding competition must be completed individually through the designated Google Form provided by the competition organizers. Each registered participant/team is required to provide the necessary information and details accurately in the form.
9. **Submission Limitation:** Each team is allowed to submit a single submission per track. Multiple submissions from the same team for a single track will not be considered.
10. **Codalab Validation Server:** The Codalab validation server is freely available for participants to use for testing and evaluation purposes. Its usage is not bound by any restrictions, as long as it complies with the competition guidelines and rules.
11. **Exceptions and Rule Modifications:** The competition organizers reserve the right to grant exceptions to any of the aforementioned rules or modify the rules in the event of abuse, technical issues, or other unforeseen circumstances. Any changes or exceptions will be communicated to the participants in a fair and transparent manner.

By submitting a project as an individual or a team, you agree to the aforementioned distribution of prizes in the case of a team submission.
Please note that the competition organizers hold no liability or responsibility for any disputes or disagreements arising within a team regarding the distribution of prizes.

## Acknowledgements
The toolkit was largely adapted from the [NTIRE22_BURSTSR](https://github.com/goutamgmb/NTIRE22_BURSTSR) challenge repository.

The toolkit uses the forward and inverse camera pipeline code from [unprocessing](https://github.com/timothybrooks/unprocessing).
