#!/bin/bash

python train_decoder.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --json_path test_dataset/data.json \
    --tags_csv_path test_dataset/tags.csv \
    --output_dir decoder_checkpoint \
    --resolution 1024 \
    --train_batch_size 4 \
    --num_epochs 15 \
    --use_bucketing \
    --base_resolution 512 \
    --max_resolution 1024 \
    --bucket_step 64 \
    --use_focal_loss \
    --use_class_balanced \
    --learning_rate 0.001 \
    --attention_heads 8
