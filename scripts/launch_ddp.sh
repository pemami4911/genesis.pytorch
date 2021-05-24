#!/bin/bash

SEED=9013
DDP_PORT=29501
cd ..
python -m torch.distributed.launch --nproc_per_node=1 train_genesis.py with experiments/train/textured_sprites/GENESIS_simple_BG.json sequential_dataset.data_path=$1 sequential_dataset.h5_path=$2 seed=$SEED training.batch_size=24 training.run_suffix=genesis-textured-simple-ln-0.7-seed=$SEED training.DDP_port=$DDP_PORT training.out_dir=/blue/ranka/pemami/neurips2021/experiments
