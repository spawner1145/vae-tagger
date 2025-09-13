import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import os
import random
import math
from diffusers.models import AutoencoderKL
from pathlib import Path

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        # spatial attn
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        # spatial attn
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.spatial_att(spatial_concat)
        return x * spatial_att

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        # 转成序列
        x_flat = x.view(batch_size, channels, seq_len).transpose(1, 2)  # (batch, seq_len, channels)
        # 残差
        residual = x_flat
        x_flat = self.norm(x_flat)
        # Q, K, V
        q = self.q_proj(x_flat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 缩放点积
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 注意力权重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # 投影和残差连接
        output = self.out_proj(attn_output) + residual
        # 重塑回原来的形状
        output = output.transpose(1, 2).view(batch_size, channels, height, width)
        
        return output

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, query_dim)
        
    def forward(self, query, key_value):
        batch_size = query.shape[0]
        
        q = self.q_proj(query).unsqueeze(1)  # (batch, 1, embed_dim)
        k = self.k_proj(key_value)  # (batch, spatial_dim, embed_dim)
        v = self.v_proj(key_value)  # (batch, spatial_dim, embed_dim)
        
        # mha
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
        
        output = self.out_proj(attn_output.squeeze(1))
        return output + query  # 残差连接

def get_image_transform(resolution, use_bucketing=False, aspect_ratio_bucket=None):
    if use_bucketing and aspect_ratio_bucket is not None:
        target_width, target_height = aspect_ratio_bucket
        return transforms.Compose([
            SmartResize(target_width, target_height),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        # 传统的正方形resize（会变形）
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

class SmartResize:
    """保持长宽比的同时裁剪到目标尺寸"""
    def __init__(self, target_width, target_height, crop_mode='center'):
        self.target_width = target_width
        self.target_height = target_height
        self.crop_mode = crop_mode  # 'center', 'random', 'top', 'bottom'
    
    def __call__(self, img):
        original_width, original_height = img.size
        target_ratio = self.target_width / self.target_height
        original_ratio = original_width / original_height
        
        if original_ratio > target_ratio:
            new_width = int(original_height * target_ratio)
            new_height = original_height
            
            if self.crop_mode == 'center':
                left = (original_width - new_width) // 2
            elif self.crop_mode == 'random':
                left = random.randint(0, original_width - new_width)
            else:
                left = 0
                
            img = img.crop((left, 0, left + new_width, new_height))
        elif original_ratio < target_ratio:
            new_width = original_width
            new_height = int(original_width / target_ratio)
            
            if self.crop_mode == 'center':
                top = (original_height - new_height) // 2
            elif self.crop_mode == 'random':
                top = random.randint(0, original_height - new_height)
            else:
                top = 0
                
            img = img.crop((0, top, new_width, top + new_height))
        return img.resize((self.target_width, self.target_height), Image.LANCZOS)

class AspectRatioBucketing:
    def __init__(self, base_resolution=512, max_resolution=1024, bucket_step=64):
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.bucket_step = bucket_step
        self.buckets = self._generate_buckets()
        self.image_buckets = {}
    
    def _generate_buckets(self):
        buckets = []
        min_dim = self.base_resolution
        max_dim = self.max_resolution
        for width in range(min_dim, max_dim + 1, self.bucket_step):
            for height in range(min_dim, max_dim + 1, self.bucket_step):
                if width * height <= self.max_resolution * self.max_resolution:
                    buckets.append((width, height))
        
        return sorted(buckets)
    
    def assign_bucket(self, image_path):
        try:
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                original_ratio = original_width / original_height
                
                best_bucket = None
                min_ratio_diff = float('inf')
                
                for width, height in self.buckets:
                    bucket_ratio = width / height
                    ratio_diff = abs(bucket_ratio - original_ratio)
                    
                    if ratio_diff < min_ratio_diff:
                        min_ratio_diff = ratio_diff
                        best_bucket = (width, height)
                
                self.image_buckets[image_path] = best_bucket
                return best_bucket
                
        except Exception as e:
            print(f"警告: 无法分析图像 {image_path}: {e}")
            # 返回默认的正方形桶
            return (self.base_resolution, self.base_resolution)
    
    def get_bucket_statistics(self):
        bucket_counts = {}
        for bucket in self.image_buckets.values():
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        
        return bucket_counts
    
    def print_bucket_info(self):
        bucket_stats = self.get_bucket_statistics()
        print("长宽比分桶统计")
        print(f"总共生成 {len(self.buckets)} 个桶")
        print(f"实际使用 {len(bucket_stats)} 个桶")
        print("\n使用的桶分布")
        
        for bucket, count in sorted(bucket_stats.items(), key=lambda x: x[1], reverse=True):
            width, height = bucket
            ratio = width / height
            percentage = (count / len(self.image_buckets)) * 100
            print(f"{width}x{height} (比例{ratio:.2f}): {count} 张图像 ({percentage:.1f}%)")

def get_vae_latent_info(resolution, latent_channels=16):
    downsample_factor = 8
    latent_height = resolution // downsample_factor
    latent_width = resolution // downsample_factor
    
    return {
        'latent_channels': latent_channels,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'total_dim': latent_channels * latent_height * latent_width
    }

def get_vae_config(resolution, use_quant_conv, use_post_quant_conv):
    return {
        "in_channels": 3, "out_channels": 3, "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4, "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2, "act_fn": "silu", "latent_channels": 16, "norm_num_groups": 32,
        "sample_size": resolution, "mid_block_add_attention": True, "use_quant_conv": use_quant_conv,
        "use_post_quant_conv": use_post_quant_conv
    }

def get_image_paths(path):
    image_paths = []
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    
    if os.path.isdir(path):
        for ext in supported_extensions:
            image_paths.extend(Path(path).rglob(f'*{ext}'))
            image_paths.extend(Path(path).rglob(f'*{ext.upper()}'))
    elif os.path.isfile(path):
        if any(path.lower().endswith(ext) for ext in supported_extensions):
            image_paths.append(Path(path))
        else:
            print(f"警告: {path} 不是支持的图像格式")
    else:
        print(f"错误: 路径 {path} 不存在")
    
    return image_paths

class VAE(nn.Module):
    def __init__(self, vae_config):
        super().__init__()
        self.vae = AutoencoderKL(**vae_config)

    def forward(self, x):
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        reconstruction = self.vae.decode(z).sample
        return reconstruction, posterior

    def encode(self, x):
        # 只返回latent，用于下游任务
        return self.vae.encode(x).latent_dist.mode()

class ClassificationDecoder(nn.Module):
    def __init__(self, latent_channels, latent_height, latent_width, num_classes, use_adaptive_pooling=True):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.use_adaptive_pooling = use_adaptive_pooling
        self._debug_once = False  # 调试标记
        
        if use_adaptive_pooling:
            # 自适应池化
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 池化到固定大小
            pooled_dim = latent_channels * 16  # 4 * 4
        else:
            # 直接展平
            pooled_dim = latent_channels * latent_height * latent_width
        
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, latent_vectors):
        batch_size = latent_vectors.size(0)
        
        if self.use_adaptive_pooling:
            # 使用自适应池化
            pooled = self.adaptive_pool(latent_vectors)
            flattened = pooled.reshape(batch_size, -1)
        else:
            # 直接展平
            flattened = latent_vectors.reshape(batch_size, -1)
        
        if hasattr(self, '_debug_once') and not self._debug_once:
            print(f"latent_vectors shape: {latent_vectors.shape}")
            print(f"flattened shape: {flattened.shape}")
            print(f"expected input dim: {self.classifier[0].in_features}")
            self._debug_once = True
        
        return self.classifier(flattened)

    def get_confidence(self, latent_vectors):
        with torch.no_grad():
            logits = self(latent_vectors)
            confidences = torch.sigmoid(logits)
            sorted_confidences, indices = torch.sort(confidences, descending=True)
        return sorted_confidences, indices

class AttentionClassificationDecoder(nn.Module):
    def __init__(self, latent_channels, latent_height, latent_width, num_classes, 
                 use_spatial_attention=True, use_self_attention=True, use_cross_attention=False,
                 attention_heads=8, attention_dropout=0.1):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.use_spatial_attention = use_spatial_attention
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self._debug_once = False
        
        # spatial attn
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(latent_channels)
            print("启用spatial attn机制")
        
        # 特征维度压缩
        self.feature_compress = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(latent_channels // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # 统一到8x8
        )
        
        compressed_dim = (latent_channels // 2) * 64  # 8 * 8

        if use_self_attention:
            self.self_attention_post = MultiHeadSelfAttention(
                embed_dim=latent_channels // 2, num_heads=attention_heads, dropout=attention_dropout
            )
            print("启用多头self attn机制（8x8压缩后）")
        
        # cross attn（可选）
        if use_cross_attention:
            self.cross_attention = CrossAttention(
                query_dim=512, key_dim=compressed_dim, embed_dim=256, num_heads=attention_heads
            )
            print("启用cross attn机制")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(compressed_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(256, num_classes)
        )
        
        # 用于cross attn的查询生成器（如果启用）
        if use_cross_attention:
            self.query_generator = nn.Linear(compressed_dim, 512)

    def forward(self, latent_vectors):
        batch_size = latent_vectors.size(0)
        x = latent_vectors
        if not self._debug_once:
            print(f"输入特征形状: {x.shape}")
        
        # spatial attn
        if self.use_spatial_attention:
            x = self.spatial_attention(x)
            if not self._debug_once:
                print(f"spatial attn后: {x.shape}")
        
        # 特征压缩
        x = self.feature_compress(x)
        if not self._debug_once:
            print(f"特征压缩后: {x.shape}")

        # 在8x8上self attn
        if self.use_self_attention:
            x = self.self_attention_post(x)
            if not self._debug_once:
                print(f"post self attn后: {x.shape}")
        
        # 展平特征
        flattened = x.reshape(batch_size, -1)
        
        # cross attn
        if self.use_cross_attention:
            # 生成查询向量
            query = self.query_generator(flattened)
            # 重塑为空间形式用于cross attn
            spatial_features = x.view(batch_size, x.size(1), -1).transpose(1, 2)
            # 应用cross attn
            attended_query = self.cross_attention(query, spatial_features)
            # 使用注意力结果
            flattened = flattened + attended_query.mean(dim=1, keepdim=True).expand_as(flattened)
        
        # 最终分类
        output = self.classifier(flattened)
        
        if not self._debug_once:
            print(f"最终输出: {output.shape}")
            self._debug_once = True
        
        return output
    
    def get_confidence(self, latent_vectors):
        with torch.no_grad():
            logits = self(latent_vectors)
            confidences = torch.sigmoid(logits)
            sorted_confidences, indices = torch.sort(confidences, descending=True)
        return sorted_confidences, indices
    
    def get_attention_maps(self, latent_vectors):
        """获取注意力权重图，用于可视化分析"""
        attention_maps = {}
        
        if self.use_spatial_attention:
            # 开摆(
            pass
            
        return attention_maps

class TaggedImageDataset(Dataset):
    def __init__(self, json_path, tags_csv_path, transform=None, use_bucketing=False, 
                 base_resolution=512, max_resolution=1024, bucket_step=64):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tags_df = pd.read_csv(tags_csv_path)
        self.tags = list(self.tags_df['name'])
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.tags)}
        self.idx_to_tag = {i: tag for tag, i in self.tag_to_idx.items()}
        self.transform = transform
        self.image_paths = list(self.data.keys())
        
        # 分桶
        self.use_bucketing = use_bucketing
        if use_bucketing:
            print("启用长宽比分桶功能...")
            self.bucketing = AspectRatioBucketing(base_resolution, max_resolution, bucket_step)
            self._analyze_images()
            self.bucketing.print_bucket_info()
        else:
            self.bucketing = None

        self.image_labels = {}
        for path, prompt in self.data.items():
            labels = torch.zeros(len(self.tags), dtype=torch.float32)
            if ',' in prompt:
                # 多标签格式tag1:1.0, tag2:0.8(可以有权重)
                tag_entries = [t.strip() for t in prompt.split(',')]
                for entry in tag_entries:
                    if ':' in entry:
                        tag, weight = entry.split(':', 1)
                        tag = tag.strip()
                        try:
                            weight = float(weight.strip())
                        except ValueError:
                            weight = 1.0
                    else:
                        tag = entry.strip()
                        weight = 1.0
                    
                    if tag in self.tag_to_idx:
                        labels[self.tag_to_idx[tag]] = weight
            else:
                # 单标签格式也可以包含权重
                if ':' in prompt:
                    tag, weight = prompt.split(':', 1)
                    tag = tag.strip()
                    try:
                        weight = float(weight.strip())
                    except ValueError:
                        weight = 1.0
                else:
                    tag = prompt.strip()
                    weight = 1.0
                
                if tag in self.tag_to_idx:
                    labels[self.tag_to_idx[tag]] = weight
                    
            self.image_labels[path] = labels
            
        # 打印数据集标签分布统计
        #self._print_label_distribution()
    
    def _analyze_images(self):
        print("正在分析图像分辨率...")
        for image_path in self.image_paths:
            self.bucketing.assign_bucket(image_path)
            
    def _print_label_distribution(self):
        label_counts = {}
        label_combinations = {}
        total_labels_per_image = []
        
        for path, labels in self.image_labels.items():
            active_labels = []
            for i, label_val in enumerate(labels):
                if label_val > 0:
                    tag_name = self.idx_to_tag[i]
                    label_counts[tag_name] = label_counts.get(tag_name, 0) + 1
                    active_labels.append(tag_name)
            total_labels_per_image.append(len(active_labels))
            if 2 <= len(active_labels) <= 3:
                combo = tuple(sorted(active_labels))
                label_combinations[combo] = label_combinations.get(combo, 0) + 1
        
        print("数据集标签分布")
        print(f"总计: {len(self.image_paths)} 张图像, {len(label_counts)} 个不同标签")
        
        # 单标签统计
        #print("\n各标签出现频次")
        #for tag, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        #    percentage = (count / len(self.image_paths)) * 100
        #    print(f"{tag}: {count} 张图像 ({percentage:.1f}%)")
        
        # 多标签统计
        #avg_labels = sum(total_labels_per_image) / len(total_labels_per_image)
        #max_labels = max(total_labels_per_image) if total_labels_per_image else 0
        #min_labels = min(total_labels_per_image) if total_labels_per_image else 0
        #print(f"\n多标签统计")
        #print(f"平均每张图像标签数: {avg_labels:.2f}")
        #print(f"标签数范围: {min_labels} - {max_labels}")
        
        # 显示最常见的标签组合
        #if label_combinations:
        #    print("\n常见标签组合 (前10个)")
        #    sorted_combos = sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)[:10]
        #    for combo, count in sorted_combos:
        #        print(f"{', '.join(combo)}: {count} 张图像")
    
    def _online_triplet_mining(self, anchor_path, anchor_labels, max_candidates=100):
        """在线三元组挖掘"""
        anchor_tag_count = anchor_labels.sum().item()
        positive_paths = []
        negative_paths = []
        
        # 随机采样一部分候选样本进行比较
        candidate_paths = random.sample(
            [p for p in self.image_paths if p != anchor_path], 
            min(max_candidates, len(self.image_paths) - 1)
        )
        
        for other_path in candidate_paths:
            overlap = (self.image_labels[other_path] * anchor_labels).sum().item()
            if overlap > 0:
                positive_paths.append(other_path)
            else:
                negative_paths.append(other_path)
        
        return positive_paths, negative_paths
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_labels = self.image_labels[anchor_path]
        anchor_img = self._load_and_transform(anchor_path)
        positive_path, negative_path = self._sample_triplet_paths(anchor_path, anchor_labels)
        
        positive_img = self._load_and_transform(positive_path)
        negative_img = self._load_and_transform(negative_path)

        return {
            "pixel_values": anchor_img,
            "labels": anchor_labels,
            "anchor": anchor_img,
            "positive": positive_img,
            "negative": negative_img,
        }
    
    def _sample_triplet_paths(self, anchor_path, anchor_labels):
        anchor_tag_count = anchor_labels.sum().item()
        positive_paths, negative_paths = self._online_triplet_mining(anchor_path, anchor_labels)
        if anchor_tag_count > 1 and positive_paths:
            positive_scores = []
            for path in positive_paths:
                overlap = (self.image_labels[path] * anchor_labels).sum().item()
                positive_scores.append((path, overlap))
            positive_scores.sort(key=lambda x: x[1], reverse=True)
            if random.random() < 0.7 and len(positive_scores) > 1:
                positive_path = positive_scores[0][0]
            else:
                positive_path = random.choice(positive_paths)
        elif positive_paths:
            positive_path = random.choice(positive_paths)
        else:
            # 如果找不到正样本，使用anchor本身（对于稀少标签组合）
            positive_path = anchor_path
            if anchor_tag_count == 1:
                print(f"警告: 单标签稀少，未找到正样本: {anchor_path}")
            else:
                print(f"警告: 多标签组合稀少({int(anchor_tag_count)}个标签)，未找到正样本: {anchor_path}")
        # 选择负样本
        if negative_paths:
            negative_path = random.choice(negative_paths)
        else:
            # 如果找不到负样本，随机选择一个不同的图像
            possible_paths = [p for p in self.image_paths if p != anchor_path]
            negative_path = random.choice(possible_paths) if possible_paths else anchor_path
            
        return positive_path, negative_path
        
    def _load_and_transform(self, path):
        try:
            img = Image.open(path).convert("RGB")
            
            if self.use_bucketing and self.bucketing:
                bucket = self.bucketing.image_buckets.get(path)
                if bucket:
                    # 动态分桶resize
                    width, height = bucket
                    smart_transform = transforms.Compose([
                        SmartResize(width, height, crop_mode='center'),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
                    return smart_transform(img)
                elif self.transform:
                    return self.transform(img)
                else:
                    default_transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
                    return default_transform(img)
            else:
                return self.transform(img) if self.transform else img
                
        except Exception as e:
            print(f"警告: 无法加载图像 {path}: {e}")
            # 返回一个黑色图像作为占位符
            if self.use_bucketing:
                # 使用默认的正方形尺寸
                dummy_img = Image.new('RGB', (512, 512), (0, 0, 0))
            else:
                dummy_img = Image.new('RGB', (224, 224), (0, 0, 0))
                
            if self.transform:
                return self.transform(dummy_img)
            else:
                return dummy_img

def create_attention_decoder(latent_channels, latent_height, latent_width, num_classes, 
                           attention_config=None):
    if attention_config is None:
        print("使用标准分类解码器")
        return ClassificationDecoder(latent_channels, latent_height, latent_width, num_classes)
    
    print("使用注意力增强分类解码器")
    return AttentionClassificationDecoder(
        latent_channels=latent_channels,
        latent_height=latent_height, 
        latent_width=latent_width,
        num_classes=num_classes,
        use_spatial_attention=attention_config.get('use_spatial_attention', True),
        use_self_attention=attention_config.get('use_self_attention', True),
        use_cross_attention=attention_config.get('use_cross_attention', False),
        attention_heads=attention_config.get('attention_heads', 8),
        attention_dropout=attention_config.get('attention_dropout', 0.1)
    )