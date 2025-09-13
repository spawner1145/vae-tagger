#!/bin/bash

python infer_full.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --decoder_checkpoint decoder_checkpoint/best_pytorch_model.bin \
    --image_path test_dataset/images/synthetic_0001.jpg \
    --tags_csv_path test_dataset/tags.csv \
    --output_dir single_inference \
    --confidence_threshold 0.3
