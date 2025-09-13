#!/bin/bash

python vae_reconstruction_test.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --output_dir vae_reconstruction_output \
    --resolution 512 \
    --show_result