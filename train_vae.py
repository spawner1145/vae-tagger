import argparse
import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from torchvision import transforms
from modules import TaggedImageDataset, get_image_transform
from improved_losses import ImprovedTripletLoss, SimplifiedCombinedLoss
from diffusers_vae_loader import (
    load_diffusers_vae_from_config, 
    DiffusersVAEWrapper, 
    create_vae_from_config_file,
    get_diffusers_vae_config
)
from torch.optim import AdamW
import json

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_vae(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.seed is not None:
        set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_dir=args.output_dir)
    if args.vae_config_path and os.path.exists(args.vae_config_path):
        print(f"从配置文件创建VAE: {args.vae_config_path}")
        model = create_vae_from_config_file(args.vae_config_path, args.vae_checkpoint)
    elif args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        print(f"直接加载预训练VAE模型: {args.vae_checkpoint}")
        vae_config = get_diffusers_vae_config()
        vae_model = load_diffusers_vae_from_config(vae_config, args.vae_checkpoint)
        model = DiffusersVAEWrapper(vae_model)
    else:
        print("使用默认配置创建新的VAE模型")
        vae_config = get_diffusers_vae_config()
        vae_config["sample_size"] = args.resolution
        vae_model = load_diffusers_vae_from_config(vae_config)
        model = DiffusersVAEWrapper(vae_model)
    
    if args.enable_xformers_memory_efficient_attention:
        from xformers.ops import memory_efficient_attention
        model.vae.enable_xformers_memory_efficient_attention()
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
    
    total_size = len(dataset)
    val_size = max(1, int(total_size * 0.1))  # 10% 作为验证集
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=max(1, args.num_workers // 2),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 三元组损失
    triplet_loss_fn = ImprovedTripletLoss(margin=args.triplet_margin, similarity_type=args.similarity_type)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, 
        optimizer=optimizer, 
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_epochs * len(train_dataloader)
    )
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # 训练循环
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        
        for step, batch in enumerate(train_dataloader):
            anchor = batch["anchor"].to(accelerator.device)
            positive = batch["positive"].to(accelerator.device)
            negative = batch["negative"].to(accelerator.device)
            anchor_labels = batch["labels"].to(accelerator.device)
            positive_labels = batch.get("positive_labels", anchor_labels).to(accelerator.device)
            
            reconstruction_a, posterior_a = model(anchor)
            _, posterior_p = model(positive)
            _, posterior_n = model(negative)
            
            z_a = posterior_a.sample()
            z_p = posterior_p.sample()
            z_n = posterior_n.sample()
            
            if args.use_simplified_vae_loss:
                recon_loss = F.mse_loss(reconstruction_a, anchor)
                triplet_loss = triplet_loss_fn(
                    z_a.reshape(z_a.size(0), -1), 
                    z_p.reshape(z_p.size(0), -1), 
                    z_n.reshape(z_n.size(0), -1),
                    anchor_labels,
                    positive_labels
                )
                total_loss = (args.reconstruction_weight * recon_loss + 
                             args.triplet_weight * triplet_loss)
                
                kl_raw = (posterior_a.kl() + posterior_p.kl() + posterior_n.kl()) / 3
                if len(kl_raw.shape) > 1:
                    kl_mean = kl_raw.mean()
                else:
                    kl_mean = kl_raw.mean()
                kl_loss_for_log = torch.log(1 + kl_mean / 10000)
            else:
                recon_loss = F.mse_loss(reconstruction_a, anchor)
                kl_raw = (posterior_a.kl() + posterior_p.kl() + posterior_n.kl()) / 3
                if len(kl_raw.shape) > 1:
                    kl_mean = kl_raw.mean()
                else:
                    kl_mean = kl_raw.mean()

                kl_loss = torch.log(1 + kl_mean / 10000)
                
                triplet_loss = triplet_loss_fn(
                    z_a.reshape(z_a.size(0), -1), 
                    z_p.reshape(z_p.size(0), -1), 
                    z_n.reshape(z_n.size(0), -1),
                    anchor_labels,
                    positive_labels
                )
                total_loss = (args.reconstruction_weight * recon_loss + 
                             args.kl_weight * kl_loss + 
                             args.triplet_weight * triplet_loss)
                kl_loss_for_log = kl_loss
            
            accelerator.backward(total_loss)

            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss_sum += total_loss.item()
            train_steps += 1

            current_lr = optimizer.param_groups[0]['lr']
            
            if accelerator.is_main_process and step % args.logging_steps == 0:
                avg_loss = train_loss_sum / train_steps
                
                if args.use_simplified_vae_loss:
                    print(
                        f"Epoch: {epoch}, Step: {step}, "
                        f"Total: {total_loss.item():.4f}, "
                        f"Recon: {recon_loss.item():.4f}, "
                        f"Triplet: {triplet_loss.item():.4f}, "
                        f"KL(监控): {kl_loss_for_log.item():.4f}, "
                        f"LR: {current_lr:.2e}"
                    )
                else:
                    print(
                        f"Epoch: {epoch}, Step: {step}, "
                        f"Total: {total_loss.item():.4f}, "
                        f"Recon: {recon_loss.item():.4f}, "
                        f"KL: {kl_loss_for_log.item():.4f}, "
                        f"Triplet: {triplet_loss.item():.4f}, "
                        f"LR: {current_lr:.2e}"
                    )
        
        # 验证
        val_loss = 0.0
        val_steps = 0
        model.eval()
        
        with torch.no_grad():
            for batch in val_dataloader:
                anchor = batch["anchor"].to(accelerator.device)
                positive = batch["positive"].to(accelerator.device)
                negative = batch["negative"].to(accelerator.device)
                anchor_labels = batch["labels"].to(accelerator.device)
                positive_labels = batch.get("positive_labels", anchor_labels).to(accelerator.device)
                
                reconstruction_a, posterior_a = model(anchor)
                _, posterior_p = model(positive)
                _, posterior_n = model(negative)
                
                z_a = posterior_a.sample()
                z_p = posterior_p.sample()
                z_n = posterior_n.sample()
                
                if args.use_simplified_vae_loss:
                    recon_loss = F.mse_loss(reconstruction_a, anchor)
                    triplet_loss = triplet_loss_fn(
                        z_a.reshape(z_a.size(0), -1), 
                        z_p.reshape(z_p.size(0), -1), 
                        z_n.reshape(z_n.size(0), -1),
                        anchor_labels,
                        positive_labels
                    )
                    total_loss = (args.reconstruction_weight * recon_loss + 
                                 args.triplet_weight * triplet_loss)
                else:
                    recon_loss = F.mse_loss(reconstruction_a, anchor)
                    kl_raw = (posterior_a.kl() + posterior_p.kl() + posterior_n.kl()) / 3
                    if len(kl_raw.shape) > 1:
                        kl_mean = kl_raw.mean()
                    else:
                        kl_mean = kl_raw.mean()
                    kl_loss = torch.log(1 + kl_mean / 10000)
                    
                    triplet_loss = triplet_loss_fn(
                        z_a.reshape(z_a.size(0), -1), 
                        z_p.reshape(z_p.size(0), -1), 
                        z_n.reshape(z_n.size(0), -1),
                        anchor_labels,
                        positive_labels
                    )
                    total_loss = (args.reconstruction_weight * recon_loss + 
                                 args.kl_weight * kl_loss + 
                                 args.triplet_weight * triplet_loss)
                
                val_loss += total_loss.item()
                val_steps += 1
        
        avg_train_loss = train_loss_sum / train_steps
        avg_val_loss = val_loss / val_steps
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['learning_rates'].append(current_lr)

        if accelerator.is_main_process:
            print(f"Epoch {epoch} completed - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
                accelerator.save_state(os.path.join(args.output_dir, "best_checkpoint"))
                unwrapped_model = accelerator.unwrap_model(model)
                vae_save_path = os.path.join(args.output_dir, "best_vae")
                os.makedirs(vae_save_path, exist_ok=True)
                unwrapped_model.vae.save_pretrained(vae_save_path)
                print(f"最佳模型已保存到: {vae_save_path}")
            if (epoch + 1) % args.save_steps == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
                unwrapped_model = accelerator.unwrap_model(model)
                vae_save_path = os.path.join(args.output_dir, f"vae_checkpoint_epoch_{epoch}")
                os.makedirs(vae_save_path, exist_ok=True)
                unwrapped_model.vae.save_pretrained(vae_save_path)
                print(f"检查点已保存到: {vae_save_path}")

    if accelerator.is_main_process:
        print("训练完成，保存训练历史...")
        with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
            json.dump(training_history, f, indent=2)
        print("VAE训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", type=str, default=None, 
                       help="预训练VAE模型文件路径 (.safetensors)")
    parser.add_argument("--vae_config_path", type=str, default=None,
                       help="VAE配置文件路径 (JSON格式)")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--tags_csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="vae_output")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    
    # 损失权重参数
    parser.add_argument("--use_simplified_vae_loss", action="store_true", default=True, help="使用简化的VAE损失（重建+三元组）")
    parser.add_argument("--reconstruction_weight", type=float, default=0.01)
    parser.add_argument("--kl_weight", type=float, default=1e-2, help="KL散度损失权重（log变换后）")
    parser.add_argument("--triplet_weight", type=float, default=1.0)
    parser.add_argument("--triplet_margin", type=float, default=1.0)
    parser.add_argument("--similarity_type", type=str, default="cosine", choices=["cosine", "euclidean"], help="三元组损失的相似性度量")
    
    # 训练优化参数
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="学习率预热步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_steps", type=int, default=5, help="模型保存间隔（epochs）")
    
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="启用 xformers 加速")
    
    # 分桶功能参数
    parser.add_argument("--use_bucketing", action="store_true", help="启用长宽比分桶功能")
    parser.add_argument("--base_resolution", type=int, default=512, help="分桶的基础分辨率")
    parser.add_argument("--max_resolution", type=int, default=1024, help="分桶的最大分辨率")
    parser.add_argument("--bucket_step", type=int, default=64, help="分桶的步长")

    # 数据加载/稳定性
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作线程数")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 预取倍数（每个worker）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="启用cudnn benchmark以提高卷积性能（输入尺寸固定时建议开启）")
    parser.add_argument("--cudnn_deterministic", action="store_true", help="启用确定性cuDNN以提高复现性（可能降低性能）")
    
    args = parser.parse_args()
    train_vae(args)