#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3  python -m torch.distributed.launch --nproc_per_node=2 train_timm.py --config data/cifar100.yml --model stmixerv3 --seed 40