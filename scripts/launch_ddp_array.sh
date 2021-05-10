#!/bin/bash

#SEEDS=(9725 9013 1600)
SEEDS=(1)
DDP_PORT=29520
cd ..

pwd 

for SEED in ${SEEDS[@]}; do
    python -m torch.distributed.launch --nproc_per_node=2 train_genesis.py with experiments/train/clevr6/GENESIS.json seed=$SEED training.batch_size=16 training.run_suffix=genesis-clevr6-debug-seed=$SEED training.DDP_port=$DDP_PORT dataset.data_path=$1 dataset.h5_path=$2 
done
