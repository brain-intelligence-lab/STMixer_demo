# STMixer_demo
Code for paper: Spiking Token Mixer: An Event-Driven Friendly Former Structure for Spiking Neural Networks


## Prerequisites
The Following Setup is tested and it is working:
 * Python>=3.5
 * Pytorch>=1.9.0
 * Cuda>=10.2

## Description
* We use Dspike surrogate gradient to realize the backward of step function.
* LIF model is build in LIFSpike in models/layer.py.
* You can use the following code to simply run this demo for cifar100:
  ```
 CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2 train_timm.py --config data/cifar100.yml --model stmixerv3 --seed 40
 ```
* Please change the relevant code on lines 834-836 of train_timm.py to change the network hyperparameter.

## Pre-trained models
* The STMixer_8_768 models we used on ImageNet are avilable [here](https://drive.google.com/file/d/12VeoDWvnUcPk-8uM7gP9XRQSSzb5O5D-/view?usp=drive_link).
* 

## Citation
```
@inproceedings{
anonymous2024spiking,
title={Spiking Token Mixer:  A event-driven friendly Former structure for spiking neural networks},
author={Anonymous},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=iYcY7KAkSy}
}
```
