#!/bin/bash

SEED=9725
DDP_PORT=29501
cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_genesis.py with experiments/train/textured_sprites/GENESIS_simple_BG.json sequential_dataset.data_path=/blue/ranka/pemami/neurips2021 seed=$SEED training.batch_size=8 training.run_suffix=genesis-textured-simple-bg-3-seed=$SEED training.DDP_port=$DDP_PORT training.out_dir=/blue/ranka/pemami/neurips2021/experiments
