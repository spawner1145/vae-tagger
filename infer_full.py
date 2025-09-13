import argparse
import torch
import os
import json
from PIL import Image
import pandas as pd
from pathlib import Path
from modules import ClassificationDecoder, create_attention_decoder, get_image_transform, get_image_paths, get_vae_latent_info
from diffusers_vae_loader import (
    load_diffusers_vae_from_config, 
    DiffusersVAEWrapper, 
    create_vae_from_config_file,
    get_diffusers_vae_config
)

def load_models(args, device="cuda"):
    if args.vae_config_path and os.path.exists(args.vae_config_path):
        print(f"从配置文件创建VAE: {args.vae_config_path}")
        vae_model = create_vae_from_config_file(args.vae_config_path, args.vae_checkpoint)
    elif args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        print(f"直接加载预训练VAE模型: {args.vae_checkpoint}")
        vae_config = get_diffusers_vae_config()
        vae_diffusers = load_diffusers_vae_from_config(vae_config, args.vae_checkpoint)
        vae_model = DiffusersVAEWrapper(vae_diffusers)
    else:
        raise RuntimeError("必须提供 VAE 模型检查点或配置文件")
    vae_model.to(device)
    vae_model.eval()
    latent_info = get_vae_latent_info(args.resolution)
    print(f"VAE潜在空间信息: {latent_info}")
    tags_df = pd.read_csv(args.tags_csv_path)
    num_classes = len(tags_df)
    if args.use_attention:
        print("使用注意力分类解码器")
        attention_config = {
            'use_spatial_attention': getattr(args, 'use_spatial_attention', True),
            'use_self_attention': getattr(args, 'use_self_attention', True),
            'use_cross_attention': getattr(args, 'use_cross_attention', False),
            'attention_heads': getattr(args, 'attention_heads', 8)
        }
        decoder = create_attention_decoder(
            latent_channels=latent_info['latent_channels'],
            latent_height=latent_info['latent_height'], 
            latent_width=latent_info['latent_width'],
            num_classes=num_classes,
            attention_config=attention_config
        )
    else:
        print("使用标准分类解码器")
        decoder = ClassificationDecoder(
            latent_channels=latent_info['latent_channels'],
            latent_height=latent_info['latent_height'], 
            latent_width=latent_info['latent_width'],
            num_classes=num_classes,
            use_adaptive_pooling=True
        )
    
    if not os.path.exists(args.decoder_checkpoint):
        raise RuntimeError(f"解码器模型文件不存在: {args.decoder_checkpoint}")
    try:
        decoder_state_dict = torch.load(args.decoder_checkpoint, map_location='cpu')
        decoder.load_state_dict(decoder_state_dict, strict=False)
        print(f"成功加载Decoder模型: {args.decoder_checkpoint}")
    except Exception as e:
        raise RuntimeError(f"无法加载Decoder模型: {e}")
    
    decoder.to(device)
    decoder.eval()

    return vae_model, decoder, tags_df['name'].tolist()

def infer_and_classify(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vae_model, decoder, tag_names = load_models(args, device)
    
    transform = get_image_transform(args.resolution)

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图像路径未找到: {args.image_path}")

    image_paths = get_image_paths(args.image_path)

    if not image_paths:
        print("未找到任何图像文件，请检查路径。")
        return

    results = {}
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
                        sorted_confidences, indices = decoder.get_confidence(latent)
                else:
                    latent = vae_model.encode(processed_img)
                    sorted_confidences, indices = decoder.get_confidence(latent)
                predicted_tags = []
                all_predictions = []
                
                for confidence, index in zip(sorted_confidences[0], indices[0]):
                    tag_name = tag_names[index.item()]
                    conf_value = confidence.item()
                    all_predictions.append({"tag": tag_name, "confidence": conf_value})
                    
                    if conf_value >= args.confidence_threshold:
                        predicted_tags.append({
                            "tag": tag_name,
                            "confidence": float(f"{conf_value:.4f}")
                        })
                results[str(img_path)] = {
                    "predicted_tags": predicted_tags,
                    "total_tags_above_threshold": len(predicted_tags),
                    "max_confidence": float(f"{max([p['confidence'] for p in all_predictions]):.4f}"),
                    "avg_confidence_top5": float(f"{sum([p['confidence'] for p in all_predictions[:5]]) / 5:.4f}")
                }
                processed_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {processed_count}/{len(image_paths)} 图像 (跳过 {error_count} 个错误)")

            except Exception as e:
                error_count += 1
                print(f"跳过图像 {img_path}，错误原因: {e}")

    print(f"处理完成！成功: {processed_count}, 失败: {error_count}, 总计: {len(image_paths)}")

    output_path = Path(args.output_dir) / "classification_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"分类结果已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用VAE和分类解码器进行图像分类。")
    parser.add_argument("--vae_checkpoint", type=str, required=True, 
                       help="预训练VAE模型文件路径 (.safetensors)")
    parser.add_argument("--vae_config_path", type=str, default=None,
                       help="VAE配置文件路径 (JSON格式)")
    parser.add_argument("--decoder_checkpoint", type=str, required=True,
                       help="Decoder模型文件路径 (.bin/.pth)")
    parser.add_argument("--image_path", type=str, required=True, help="单个图像文件或包含图像的目录")
    parser.add_argument("--tags_csv_path", type=str, required=True, help="包含所有分类头的CSV文件")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="结果保存目录")
    parser.add_argument("--resolution", type=int, default=1024, help="模型训练时的分辨率")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="置信度阈值")
    
    # 注意力机制参数
    parser.add_argument("--use_attention", action="store_true", default=True,
                       help="使用注意力机制 (默认开启)")
    parser.add_argument("--no_attention", action="store_true",
                       help="禁用注意力机制")
    parser.add_argument("--use_spatial_attention", action="store_true", default=True,
                       help="启用空间注意力")
    parser.add_argument("--use_self_attention", action="store_true", default=True,
                       help="启用自注意力")
    parser.add_argument("--use_cross_attention", action="store_true",
                       help="启用交叉注意力")
    parser.add_argument("--attention_heads", type=int, default=8,
                       help="注意力头数")
    parser.add_argument("--model_checkpoint", type=str, default=None,
                       help="(已弃用) 包含VAE和Decoder权重的父目录，将自动拆分为vae_checkpoint和decoder_checkpoint")
    
    args = parser.parse_args()
    
    # 处理注意力机制参数
    if args.no_attention:
        args.use_attention = False

    if args.model_checkpoint and (not args.vae_checkpoint or not args.decoder_checkpoint):
        print("使用向后兼容模式，从model_checkpoint参数推导VAE和Decoder路径")
        args.vae_checkpoint = args.model_checkpoint
        args.decoder_checkpoint = args.model_checkpoint
    
    infer_and_classify(args)