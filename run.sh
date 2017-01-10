#!/bin/bash

CUDA_VISIBLE_DEVICES=0 save_folder=normal th -i doall.lua
# CUDA_VISIBLE_DEVICES=3 save_folder=input_noise input_noise=1 th doall.lua
# CUDA_VISIBLE_DEVICES=3 save_folder=label_flip label_flip=1 th doall.lua
# CUDA_VISIBLE_DEVICES=3 save_folder=both input_noise=1 label_flip=1 th doall.lua