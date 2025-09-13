import os
import json
import torch
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file as load_safetensors

def load_diffusers_vae_from_config(config_dict, model_path=None):
    vae = AutoencoderKL(
        in_channels=config_dict.get("in_channels", 3),
        out_channels=config_dict.get("out_channels", 3),
        down_block_types=config_dict.get("down_block_types", [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D", 
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ]),
        up_block_types=config_dict.get("up_block_types", [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D", 
            "UpDecoderBlock2D"
        ]),
        block_out_channels=config_dict.get("block_out_channels", [128, 256, 512, 512]),
        layers_per_block=config_dict.get("layers_per_block", 2),
        act_fn=config_dict.get("act_fn", "silu"),
        latent_channels=config_dict.get("latent_channels", 16),
        norm_num_groups=config_dict.get("norm_num_groups", 32),
        sample_size=config_dict.get("sample_size", 1024),
        scaling_factor=config_dict.get("scaling_factor", 0.3611),
        shift_factor=config_dict.get("shift_factor", 0.1159),
        use_quant_conv=config_dict.get("use_quant_conv", False),
        use_post_quant_conv=config_dict.get("use_post_quant_conv", False),
        force_upcast=config_dict.get("force_upcast", True),
        mid_block_add_attention=config_dict.get("mid_block_add_attention", True)
    )
    
    if model_path and os.path.exists(model_path):
        print(f"加载预训练权重: {model_path}")
        if model_path.endswith('.safetensors'):
            state_dict = load_safetensors(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys}")
            
        print("成功加载预训练VAE权重")
    
    return vae

def load_diffusers_vae_from_pretrained(model_name_or_path, subfolder=None):
    try:
        if subfolder:
            vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder=subfolder)
        else:
            vae = AutoencoderKL.from_pretrained(model_name_or_path)
        print(f"成功从 {model_name_or_path} 加载预训练VAE")
        return vae
    except Exception as e:
        print(f"从 {model_name_or_path} 加载VAE失败: {e}")
        return None

class DiffusersVAEWrapper(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model
        
    def forward(self, x):
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        reconstruction = self.vae.decode(z).sample
        return reconstruction, posterior
    
    def encode(self, x):
        posterior = self.vae.encode(x).latent_dist
        latent = posterior.mode()
        if hasattr(self.vae.config, 'scaling_factor'):
            latent = latent * self.vae.config.scaling_factor
        if hasattr(self.vae.config, 'shift_factor'):
            latent = latent + self.vae.config.shift_factor
            
        return latent
    
    def decode(self, z):
        if hasattr(self.vae.config, 'shift_factor'):
            z = z - self.vae.config.shift_factor
        if hasattr(self.vae.config, 'scaling_factor'):
            z = z / self.vae.config.scaling_factor
            
        return self.vae.decode(z).sample

def create_vae_from_config_file(config_path, model_path=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    vae = load_diffusers_vae_from_config(config, model_path)
    return DiffusersVAEWrapper(vae)

def get_diffusers_vae_config():
    return {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.30.0.dev0",
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 16,
        "latents_mean": None,
        "latents_std": None,
        "layers_per_block": 2,
        "mid_block_add_attention": True,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
        ],
        "use_post_quant_conv": False,
        "use_quant_conv": False
    }