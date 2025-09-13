#!/bin/bash

python train_vae.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --json_path test_dataset/data.json \
    --tags_csv_path test_dataset/tags.csv \
    --output_dir vae_checkpoint \
    --resolution 1024 \
    --train_batch_size 4 \
    --num_epochs 20 \
    --use_bucketing \
    --base_resolution 512 \
    --max_resolution 1024 \
    --bucket_step 64 \
    --mixed_precision fp16 \
    --learning_rate 0.0001 \
    --use_simplified_vae_loss
