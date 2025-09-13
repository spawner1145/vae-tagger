#!/bin/bash

python infer_vae.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --image_path test_dataset/images \
    --output_dir vae_inference
