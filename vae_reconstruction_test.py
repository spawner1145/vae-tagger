"""
VAE 重建测试脚本
"""
import argparse
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from diffusers_vae_loader import (
    load_diffusers_vae_from_config, 
    DiffusersVAEWrapper, 
    create_vae_from_config_file,
    get_diffusers_vae_config
)

def load_vae_model(args, device="cuda"):
    if args.vae_config_path and os.path.exists(args.vae_config_path):
        print(f"从配置文件创建VAE: {args.vae_config_path}")
        model = create_vae_from_config_file(args.vae_config_path, args.vae_checkpoint)
    elif args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        print(f"直接加载预训练VAE模型: {args.vae_checkpoint}")
        vae_config = get_diffusers_vae_config()
        vae_diffusers = load_diffusers_vae_from_config(vae_config, args.vae_checkpoint)
        model = DiffusersVAEWrapper(vae_diffusers)
    else:
        print("使用默认配置创建新的VAE模型")
        vae_config = get_diffusers_vae_config()
        vae_config["sample_size"] = args.resolution
        vae_diffusers = load_diffusers_vae_from_config(vae_config)
        model = DiffusersVAEWrapper(vae_diffusers)
    
    model.to(device)
    model.eval()
    return model

def create_test_image(size=(512, 512)):
    width, height = size
    r = np.linspace(0, 255, width).astype(np.uint8)
    g = np.linspace(255, 0, height).astype(np.uint8)
    b = np.ones((height, width)) * 128
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        img_array[i, :, 0] = r
    for j in range(width):
        img_array[:, j, 1] = g
    img_array[:, :, 2] = b
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 6)**2
    img_array[mask] = [255, 255, 255]
    rect_size = min(width, height) // 8
    x1, y1 = center_x - rect_size, center_y - rect_size
    x2, y2 = center_x + rect_size, center_y + rect_size
    img_array[y1:y2, x1:x2] = [255, 0, 0]
    return Image.fromarray(img_array)

def load_image(image_path, target_size=(512, 512)):
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        print(f"加载图像: {image_path}")
    else:
        image = create_test_image(target_size)
        print("使用生成的测试图像")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image

def preprocess_image(image, resolution=512):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(image).unsqueeze(0)

def postprocess_image(tensor):
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.squeeze(0).cpu()
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

def test_vae_reconstruction(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    vae_model = load_vae_model(args, device)
    print("VAE 模型加载完成")
    original_image = load_image(args.image_path, (args.resolution, args.resolution))
    input_tensor = preprocess_image(original_image, args.resolution).to(device)
    print(f"输入图像形状: {input_tensor.shape}")
    with torch.no_grad():
        print("开始 VAE 编码...")
        posterior = vae_model.vae.encode(input_tensor).latent_dist
        latent = posterior.sample()
        print(f"潜在向量形状: {latent.shape}")
        print(f"潜在向量统计: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}")
        print("开始 VAE 解码...")
        reconstructed_tensor = vae_model.vae.decode(latent).sample
        print(f"重建图像形状: {reconstructed_tensor.shape}")
    reconstructed_image = postprocess_image(reconstructed_tensor)
    mse_loss = torch.nn.functional.mse_loss(input_tensor, reconstructed_tensor).item()
    print(f"重建 MSE 损失: {mse_loss:.6f}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(reconstructed_image)
    axes[1].set_title(f'VAE restruction\nMSE Loss: {mse_loss:.6f}', fontsize=14)
    axes[1].axis('off')
    diff_array = np.abs(np.array(original_image).astype(float) - np.array(reconstructed_image).astype(float))
    diff_image = diff_array / diff_array.max() if diff_array.max() > 0 else diff_array
    axes[2].imshow(diff_image)
    axes[2].set_title('difference (abs)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    comparison_path = os.path.join(args.output_dir, 'vae_reconstruction_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {comparison_path}")
    original_image.save(os.path.join(args.output_dir, 'original.png'))
    reconstructed_image.save(os.path.join(args.output_dir, 'reconstructed.png'))
    latent_path = os.path.join(args.output_dir, 'latent_vector.pt')
    torch.save(latent.cpu(), latent_path)
    print(f"潜在向量已保存到: {latent_path}")
    if args.show_result:
        plt.show()
    
    print("VAE 重建测试完成！")
    print(f"输入分辨率: {args.resolution}x{args.resolution}")
    print(f"潜在空间维度: {latent.shape}")
    print(f"压缩比: {(input_tensor.numel() / latent.numel()):.2f}:1")
    print(f"重建误差 (MSE): {mse_loss:.6f}")
    # PSNR
    psnr = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(torch.tensor(mse_loss))
    print(f"PSNR: {psnr.item():.2f} dB")

def main():
    parser = argparse.ArgumentParser(description="VAE 图片重建测试")
    
    # 模型参数
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                       help="预训练VAE模型文件路径 (.safetensors)")
    parser.add_argument("--vae_config_path", type=str, default=None,
                       help="VAE配置文件路径 (JSON格式)")
    
    # 输入输出参数
    parser.add_argument("--image_path", type=str, default=None,
                       help="输入图像路径 (可选，不提供则使用生成的测试图像)")
    parser.add_argument("--output_dir", type=str, default="vae_reconstruction_output",
                       help="输出目录")
    parser.add_argument("--resolution", type=int, default=512,
                       help="图像分辨率")
    
    # 显示参数
    parser.add_argument("--show_result", action="store_true",
                       help="显示结果图像")
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.vae_checkpoint and not args.vae_config_path:
        print("警告: 未提供VAE模型或配置，将使用默认配置创建新模型")
    
    test_vae_reconstruction(args)

if __name__ == "__main__":
    main()