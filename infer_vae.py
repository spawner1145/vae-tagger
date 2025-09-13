import argparse
import torch
import os
import json
from PIL import Image
from pathlib import Path
from modules import get_image_transform, get_image_paths
from diffusers_vae_loader import (
    load_diffusers_vae_from_config, 
    DiffusersVAEWrapper, 
    create_vae_from_config_file,
    get_diffusers_vae_config
)

def load_vae(args, device="cuda"):
    if args.vae_config_path and os.path.exists(args.vae_config_path):
        print(f"从配置文件创建VAE: {args.vae_config_path}")
        model = create_vae_from_config_file(args.vae_config_path, args.vae_checkpoint)
    elif args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        print(f"直接加载预训练VAE模型: {args.vae_checkpoint}")
        vae_config = get_diffusers_vae_config()
        vae_diffusers = load_diffusers_vae_from_config(vae_config, args.vae_checkpoint)
        model = DiffusersVAEWrapper(vae_diffusers)
    else:
        raise RuntimeError("必须提供 VAE 模型检查点或配置文件")
    
    model.to(device)
    model.eval()
    return model

def infer_and_save_latents(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vae_model = load_vae(args, device)
    
    transform = get_image_transform(args.resolution)
    
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图像路径未找到: {args.image_path}")

    image_paths = get_image_paths(args.image_path)

    if not image_paths:
        print("未找到任何图像文件，请检查路径。")
        return
        
    latent_data = {}
    processed_count = 0
    error_count = 0
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                processed_img = transform(img).unsqueeze(0).to(device)
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        latent = vae_model.encode(processed_img)
                else:
                    latent = vae_model.encode(processed_img)
                flattened_latent = latent.reshape(latent.size(0), -1).squeeze(0).cpu().numpy()
                
                latent_data[str(img_path)] = flattened_latent.tolist()
                processed_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {processed_count}/{len(image_paths)} 图像 (跳过 {error_count} 个错误)")

            except Exception as e:
                error_count += 1
                print(f"跳过图像 {img_path}，错误原因: {e}")

    print(f"处理完成！成功: {processed_count}, 失败: {error_count}, 总计: {len(image_paths)}")

    output_path = Path(args.output_dir) / "latent_vectors.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(latent_data, f, indent=4)
        
    print(f"潜在向量已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用VAE模型进行推理，输出潜在向量。")
    parser.add_argument("--vae_checkpoint", type=str, required=True, 
                       help="预训练VAE模型文件路径 (.safetensors)")
    parser.add_argument("--vae_config_path", type=str, default=None,
                       help="VAE配置文件路径 (JSON格式)")
    parser.add_argument("--image_path", type=str, required=True, help="单个图像文件或包含图像的目录")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="潜在向量保存目录")
    parser.add_argument("--resolution", type=int, default=1024, help="VAE模型训练时的分辨率")
    args = parser.parse_args()
    infer_and_save_latents(args)