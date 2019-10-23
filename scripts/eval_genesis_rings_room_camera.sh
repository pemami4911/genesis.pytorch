#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python eval_genesis.py with dataset.data_path='/data/pemami' test.checkpoint=genesis_rings_room_camera_500k.pth seed=123
