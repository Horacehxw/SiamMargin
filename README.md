# Discriminative Siamese Embedding for Object Tracking (SiamMargin)

**author:** G. Chen, L. Chen, G. Li, Y. Chen, F. Wang, S. You, C. Qian

## Notice
**Important:** The code is obtained from [VOT2019 Challenge official website](https://github.com/votchallenge/website/blob/74c8150696c54ce5c34c3e05d8ccbef52177c259/contents/vot2019/trackers.md) 

If you fail to install and run this tracker, please email the author (chenguangqi@sensetime.com)

## Prerequisites

CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz × 8 

GPU: NVIDIA GeForce GTX 1060 6GB/PCIe/SSE2

OS: ubuntu 14.04 LTS 64-bit

CUDA: 8.0

Required python version:

* python 3.6

Required packages for python 3.6:

* pytorch == 0.3.1 (cuda 80)
* numpy==1.16.2
* opencv-python==3.4.4.19
* torchvision == 0.2.1

## Install

Install the environment by executing the script `install.sh`. The pretrained checkpoint has been put into the subfolder `'./code/model.pth'`. 

Add the path of this dir to `tracker_SiamMargin.m` in the matlab toolkit workspace, like:
```matlab
tracker_command = generate_python_command('vot_SiamMargin',{'PATH/TO/THIS/DIR'});
```

Then run the test command `run_experiments.m`.