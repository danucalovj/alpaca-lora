#!/bin/bash

set WORLD_SIZE=2
set CUDA_VISIBLE_DEVICES=0,1
set OMP_NUM_THREADS=2
torchrun --nproc_per_node=2 --master_port=1234 finetune_custom.py \
--wandb_project WANDB_PROJECT_NAME \
--wandb_watch gradients \
--base_model ./base_models/BASE_MODEL \
--output_dir ./loras/LORA_NAME \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--lora_r=16 \
--data-path DATASET_NAME.json \
--num_epochs 3 \
--cutoff_len=512 \
--micro_batch_size 4 

