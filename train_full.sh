#!/bin/bash

python train_full.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --decoder_checkpoint decoder_checkpoint/best_pytorch_model.bin \
    --json_path test_dataset/data.json \
    --tags_csv_path test_dataset/tags.csv \
    --output_dir full_model \
    --resolution 1024 \
    --train_batch_size 2 \
    --num_epochs 10 \
    --use_bucketing \
    --base_resolution 512 \
    --max_resolution 1024 \
    --bucket_step 64 \
    --use_adaptive_weights \
    --use_focal_loss \
    --learning_rate 0.0001 \
    --attention_heads 8
