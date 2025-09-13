import argparse
import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from modules import ClassificationDecoder, TaggedImageDataset, get_image_transform, get_vae_latent_info, create_attention_decoder
from improved_losses import SimplifiedCombinedLoss, CombinedLoss, compute_class_distribution
from evaluation import evaluate_model, find_optimal_threshold
from diffusers_vae_loader import (
    load_diffusers_vae_from_config, 
    DiffusersVAEWrapper, 
    create_vae_from_config_file,
    get_diffusers_vae_config
)
import pandas as pd
from torch.optim import AdamW
import json

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_full(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.seed is not None:
        set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_dir=args.output_dir)
    if args.vae_config_path and os.path.exists(args.vae_config_path):
        print(f"从配置文件创建VAE: {args.vae_config_path}")
        vae_model = create_vae_from_config_file(args.vae_config_path, args.vae_checkpoint)
    elif args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        print(f"直接加载预训练VAE模型: {args.vae_checkpoint}")
        vae_config = get_diffusers_vae_config()
        vae_diffusers = load_diffusers_vae_from_config(vae_config, args.vae_checkpoint)
        vae_model = DiffusersVAEWrapper(vae_diffusers)
    else:
        print("使用默认配置创建新的VAE模型")
        vae_config = get_diffusers_vae_config()
        vae_config["sample_size"] = args.resolution
        vae_diffusers = load_diffusers_vae_from_config(vae_config)
        vae_model = DiffusersVAEWrapper(vae_diffusers)
    
    if args.enable_xformers_memory_efficient_attention:
        from xformers.ops import memory_efficient_attention
        vae_model.vae.enable_xformers_memory_efficient_attention()
    latent_info = get_vae_latent_info(args.resolution)
    print(f"VAE潜在空间信息: {latent_info}")
    tags_df = pd.read_csv(args.tags_csv_path)
    num_classes = len(tags_df['name'])
    class_names = tags_df['name'].tolist()
    if args.use_attention:
        print("启用注意力机制")
        attention_config = {
            'use_spatial_attention': args.use_spatial_attention,
            'use_self_attention': args.use_self_attention,
            'use_cross_attention': args.use_cross_attention,
            'attention_heads': args.attention_heads,
            'attention_dropout': args.attention_dropout
        }
        decoder = create_attention_decoder(
            latent_info['latent_channels'],
            latent_info['latent_height'], 
            latent_info['latent_width'],
            num_classes,
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
    if args.decoder_checkpoint and os.path.exists(args.decoder_checkpoint):
        print(f"加载预训练decoder模型: {args.decoder_checkpoint}")
        try:
            decoder_state_dict = torch.load(args.decoder_checkpoint, map_location='cpu')
            decoder.load_state_dict(decoder_state_dict, strict=False)
            print("成功加载预训练decoder模型")
        except Exception as e:
            print(f"Decoder模型加载失败，从零开始训练: {e}")
    else:
        print("从零开始训练decoder模型")

    if args.use_bucketing:
        transform = None
        print(f"启用分桶模式：基础分辨率{args.base_resolution}, 最大分辨率{args.max_resolution}")
    else:
        transform = get_image_transform(args.resolution)
        print(f"传统模式：固定分辨率{args.resolution}x{args.resolution}")
    
    dataset = TaggedImageDataset(
        json_path=args.json_path, 
        tags_csv_path=args.tags_csv_path, 
        transform=transform,
        use_bucketing=args.use_bucketing,
        base_resolution=args.base_resolution,
        max_resolution=args.max_resolution,
        bucket_step=args.bucket_step
    )
    
    class_distribution = compute_class_distribution(dataset)
    #print(f"类别分布: {dict(zip(class_names, class_distribution))}")
    
    # 分割数据集为训练和验证
    total_size = len(dataset)
    val_size = max(1, int(total_size * 0.1))  # 10% 作为验证集
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=max(1, args.num_workers // 2),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    if args.use_simplified_loss:
        print("使用简化版损失函数")
        loss_fn = SimplifiedCombinedLoss(
            classification_weight=args.bce_weight,
            triplet_weight=args.triplet_weight,
            use_focal_loss=args.use_focal_loss,
            use_class_balanced=args.use_class_balanced,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            triplet_margin=args.triplet_margin,
            similarity_type=args.similarity_type
        )
        
        # 简化版不需要adaptive weights
        params = list(vae_model.parameters()) + list(decoder.parameters())
    else:
        print("使用完整版损失函数")
        loss_fn = CombinedLoss(
            reconstruction_weight=args.reconstruction_weight,
            kl_weight=args.kl_weight,
            triplet_weight=args.triplet_weight,
            classification_weight=args.bce_weight,
            use_focal_loss=args.use_focal_loss,
            use_class_balanced=args.use_class_balanced,
            use_adaptive_weights=args.use_adaptive_weights,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            triplet_margin=args.triplet_margin,
            similarity_type=args.similarity_type
        )
        
        params = list(vae_model.parameters()) + list(decoder.parameters())
        if args.use_adaptive_weights:
            params += list(loss_fn.adaptive_weights.parameters())
    
    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, 
        optimizer=optimizer, 
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_epochs * len(train_dataloader)
    )
    
    vae_model, decoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        vae_model, decoder, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # 训练循环
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    for epoch in range(args.num_epochs):
        vae_model.train()
        decoder.train()
        train_loss_sum = 0.0
        train_steps = 0
        
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            anchor = batch["anchor"].to(accelerator.device)
            positive = batch["positive"].to(accelerator.device)
            negative = batch["negative"].to(accelerator.device)

            # VAE前向传播
            reconstruction_a, posterior_a = vae_model(anchor)
            _, posterior_p = vae_model(positive)
            _, posterior_n = vae_model(negative)
            
            z_a, z_p, z_n = posterior_a.sample(), posterior_p.sample(), posterior_n.sample()
            
            # 分类前向传播
            # 复用已算好的posterior均值作为分类输入，避免对同一batch再次encode
            with torch.no_grad():
                latent_vectors = posterior_a.mode()
                # 与 DiffusersVAEWrapper.encode 对齐缩放/平移，保证训练与推理一致
                if hasattr(vae_model.vae.config, 'scaling_factor'):
                    latent_vectors = latent_vectors * vae_model.vae.config.scaling_factor
                if hasattr(vae_model.vae.config, 'shift_factor'):
                    latent_vectors = latent_vectors + vae_model.vae.config.shift_factor
            logits = decoder(latent_vectors)
            
            # loss
            if args.use_simplified_loss:
                loss_dict = loss_fn(
                    z_a, z_p, z_n,
                    logits, labels,
                    anchor_labels=labels, 
                    positive_labels=batch.get("positive_labels", labels),
                    samples_per_class=class_distribution if args.use_class_balanced else None
                )
            else:
                loss_dict = loss_fn(
                    reconstruction_a, anchor,
                    posterior_a, posterior_p, posterior_n,
                    z_a, z_p, z_n,
                    logits, labels,
                    anchor_labels=labels, 
                    positive_labels=batch.get("positive_labels", labels),
                    samples_per_class=class_distribution if args.use_class_balanced else None
                )
            
            total_loss = loss_dict['total_loss']
            total_loss = total_loss / max(1, args.gradient_accumulation_steps)
            accelerator.backward(total_loss)
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(params, args.max_grad_norm)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            train_loss_sum += total_loss.item()
            train_steps += 1
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            if accelerator.is_main_process and step % args.logging_steps == 0:
                avg_loss = train_loss_sum / train_steps
                
                if args.use_simplified_loss:
                    print(
                        f"Epoch: {epoch}, Step: {step}, "
                        f"Loss: {total_loss.item():.4f}, "
                        f"Triplet: {loss_dict['triplet_loss'].item():.4f}, "
                        f"Class: {loss_dict['classification_loss'].item():.4f}, "
                        f"LR: {current_lr:.2e}"
                    )
                else:
                    print(
                        f"Epoch: {epoch}, Step: {step}, "
                        f"Loss: {total_loss.item():.4f}, "
                        f"Recon: {loss_dict['reconstruction_loss'].item():.4f}, "
                        f"KL: {loss_dict['kl_loss'].item():.4f}, "
                        f"Triplet: {loss_dict['triplet_loss'].item():.4f}, "
                        f"Class: {loss_dict['classification_loss'].item():.4f}, "
                        f"LR: {current_lr:.2e}"
                    )
                
                # 如果使用自适应权重，记录权重
                if not args.use_simplified_loss and args.use_adaptive_weights:
                    weights = loss_dict['adaptive_weights']
                    print(f"Adaptive weights: {weights.detach().cpu().numpy()}")
        
        # 验证
        val_loss = 0.0
        val_steps = 0
        vae_model.eval()
        decoder.eval()
        
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch["pixel_values"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)
                anchor = batch["anchor"].to(accelerator.device)
                positive = batch["positive"].to(accelerator.device) 
                negative = batch["negative"].to(accelerator.device)

                reconstruction_a, posterior_a = vae_model(anchor)
                _, posterior_p = vae_model(positive)
                _, posterior_n = vae_model(negative)
                
                z_a, z_p, z_n = posterior_a.sample(), posterior_p.sample(), posterior_n.sample()
                
                latent_vectors = posterior_a.mode()
                if hasattr(vae_model.vae.config, 'scaling_factor'):
                    latent_vectors = latent_vectors * vae_model.vae.config.scaling_factor
                if hasattr(vae_model.vae.config, 'shift_factor'):
                    latent_vectors = latent_vectors + vae_model.vae.config.shift_factor
                logits = decoder(latent_vectors)
                
                # 验证损失
                if args.use_simplified_loss:
                    loss_dict = loss_fn(
                        z_a, z_p, z_n,
                        logits, labels,
                        anchor_labels=labels, 
                        positive_labels=batch.get("positive_labels", labels),
                        samples_per_class=class_distribution if args.use_class_balanced else None
                    )
                else:
                    loss_dict = loss_fn(
                        reconstruction_a, anchor,
                        posterior_a, posterior_p, posterior_n,
                        z_a, z_p, z_n,
                        logits, labels,
                        anchor_labels=labels, 
                        positive_labels=batch.get("positive_labels", labels),
                        samples_per_class=class_distribution if args.use_class_balanced else None
                    )
                
                val_loss += loss_dict['total_loss'].item()
                val_steps += 1
        
        avg_train_loss = train_loss_sum / train_steps
        avg_val_loss = val_loss / val_steps
        
        # 记录历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['learning_rates'].append(current_lr)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} completed - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
                accelerator.save_state(os.path.join(args.output_dir, "best_checkpoint"))
                unwrapped_vae = accelerator.unwrap_model(vae_model)
                unwrapped_decoder = accelerator.unwrap_model(decoder)
                vae_output_path = os.path.join(args.output_dir, "best_vae")
                decoder_output_path = os.path.join(args.output_dir, "best_decoder")
                os.makedirs(vae_output_path, exist_ok=True)
                os.makedirs(decoder_output_path, exist_ok=True)
                # 保存 VAE
                unwrapped_vae.vae.save_pretrained(vae_output_path)
                # 保存 decoder
                torch.save(unwrapped_decoder.state_dict(), os.path.join(decoder_output_path, "pytorch_model.bin"))
                print(f"最佳VAE模型已保存到: {vae_output_path}")
                print(f"最佳decoder模型已保存到: {decoder_output_path}")
            
            # 定期保存
            if (epoch + 1) % args.save_steps == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
                unwrapped_vae = accelerator.unwrap_model(vae_model)
                unwrapped_decoder = accelerator.unwrap_model(decoder)
                vae_output_path = os.path.join(args.output_dir, "vae")
                decoder_output_path = os.path.join(args.output_dir, "decoder")
                os.makedirs(vae_output_path, exist_ok=True)
                os.makedirs(decoder_output_path, exist_ok=True)
                unwrapped_vae.vae.save_pretrained(vae_output_path)
                torch.save(unwrapped_decoder.state_dict(), os.path.join(decoder_output_path, "pytorch_model.bin"))
                
                print(f"检查点VAE模型已保存到: {vae_output_path}")
                print(f"检查点decoder模型已保存到: {decoder_output_path}")
    
    # 训练完成后评估
    if accelerator.is_main_process:
        print("训练完成，开始最终评估...")
        with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
            json.dump(training_history, f, indent=2)
        print("寻找最优分类阈值...")
        optimal_thresholds = find_optimal_threshold(
            vae_model, decoder, val_dataloader, class_names, 
            accelerator.device, args.output_dir
        )
        print("使用最优阈值进行最终评估...")
        final_metrics = evaluate_model(
            vae_model, decoder, val_dataloader, class_names,
            accelerator.device, optimal_thresholds['global_threshold'], args.output_dir
        )
        print("训练和评估完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--tags_csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="full_output")
    
    # 预训练模型路径参数
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                       help="预训练VAE模型文件路径 (.safetensors)")
    parser.add_argument("--vae_config_path", type=str, default=None,
                       help="VAE配置文件路径 (JSON格式)")
    parser.add_argument("--decoder_checkpoint", type=str, default=None,
                       help="预训练decoder模型文件路径 (.bin/.pth)")
    
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    
    # 注意力机制参数 (默认启用)
    parser.add_argument("--use_attention", action="store_true", default=True, help="启用注意力机制 (默认开启)")
    parser.add_argument("--no_attention", action="store_true", help="禁用注意力机制")
    parser.add_argument("--use_spatial_attention", action="store_true", default=True, help="启用空间注意力")
    parser.add_argument("--use_self_attention", action="store_true", default=True, help="启用自注意力") 
    parser.add_argument("--use_cross_attention", action="store_true", help="启用交叉注意力")
    parser.add_argument("--attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="注意力dropout率")
    
    # 损失权重参数
    parser.add_argument("--reconstruction_weight", type=float, default=0.01)
    parser.add_argument("--kl_weight", type=float, default=1e-7)
    parser.add_argument("--triplet_weight", type=float, default=1.0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--triplet_margin", type=float, default=1.0)
    
    # 改进的损失函数参数
    parser.add_argument("--use_simplified_loss", action="store_true", default=True, help="使用简化版损失函数（推荐）")
    parser.add_argument("--use_focal_loss", action="store_true", help="使用Focal Loss处理类别不平衡")
    parser.add_argument("--use_class_balanced", action="store_true", help="使用类别平衡损失")
    parser.add_argument("--use_adaptive_weights", action="store_true", help="使用自适应权重调整")
    parser.add_argument("--focal_alpha", type=float, default=1.0, help="Focal Loss的alpha参数")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss的gamma参数")
    parser.add_argument("--similarity_type", type=str, default="cosine", choices=["cosine", "euclidean"], help="三元组损失的相似性度量")
    
    # 训练优化参数
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="学习率预热步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_steps", type=int, default=5, help="模型保存间隔（epochs）")
    
    # 原有参数
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="启用 xformers 加速")
    parser.add_argument("--use_quant_conv", action="store_true", help="VAE config: use_quant_conv")
    parser.add_argument("--use_post_quant_conv", action="store_true", help="VAE config: use_post_quant_conv")
    
    # SafeTensors支持
    parser.add_argument("--use_safetensors", action="store_true", help="使用SafeTensors格式保存模型")
    
    # 分桶功能参数
    parser.add_argument("--use_bucketing", action="store_true", help="启用长宽比分桶功能")
    parser.add_argument("--base_resolution", type=int, default=512, help="分桶的基础分辨率")
    parser.add_argument("--max_resolution", type=int, default=1024, help="分桶的最大分辨率")
    parser.add_argument("--bucket_step", type=int, default=64, help="分桶的步长")

    # 数据加载/稳定性
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作线程数")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 预取倍数（每个worker）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数，用于小显存训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="启用cudnn benchmark以提高卷积性能（输入尺寸固定时建议开启）")
    parser.add_argument("--cudnn_deterministic", action="store_true", help="启用确定性cuDNN以提高复现性（可能降低性能）")
    
    args = parser.parse_args()
    
    # 处理注意力机制参数
    if args.no_attention:
        args.use_attention = False
        
    train_full(args)