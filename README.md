# VAE-Tagger

### 安装依赖

```bash
git clone https://github.com/spawner1145/vae-tagger.git
cd vae-tagger
pip install -r requirements.txt
```

### 1. 准备数据

```bash
# 创建测试数据集 (可选,具体数据集格式在下面)
python create_test_dataset.py --source_json your_data.json --output_dir test_dataset --test_ratio 0.1
```

### 2. 训练模型

**选项 A: 简单端到端训练 (推荐)**
```bash
python train_full.py \
    # 这边的vae的safetensors和json文件可以在 https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae 下载，开头三个都是可选参数
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --decoder_checkpoint decoder_checkpoint/best_pytorch_model.bin \
    --json_path test_dataset/data.json \
    --tags_csv_path test_dataset/tags.csv \
    --output_dir full_model \
    --resolution 1024 \
    --train_batch_size 2 \
    --num_epochs 10 \
    --use_bucketing \
    --base_resolution 512 \
    --max_resolution 1024 \
    --bucket_step 64 \
    --use_adaptive_weights \
    --use_focal_loss \
    --learning_rate 0.0001 \
    --attention_heads 8
```

**选项 B: 分步训练**
```bash
# 步骤 1: 训练 VAE
python train_vae.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --json_path test_dataset/data.json \
    --tags_csv_path test_dataset/tags.csv \
    --output_dir vae_checkpoint \
    --resolution 1024 \
    --train_batch_size 4 \
    --num_epochs 20 \
    --use_bucketing \
    --base_resolution 512 \
    --max_resolution 1024 \
    --bucket_step 64 \
    --mixed_precision fp16 \
    --learning_rate 0.0001 \
    --use_simplified_vae_loss

# 步骤 2: 训练解码器
python train_decoder.py \
    --vae_checkpoint diffusion_pytorch_model.safetensors \
    --vae_config_path diffusers_vae_config.json \
    --json_path test_dataset/data.json \
    --tags_csv_path test_dataset/tags.csv \
    --output_dir decoder_checkpoint \
    --resolution 1024 \
    --train_batch_size 4 \
    --num_epochs 15 \
    --use_bucketing \
    --base_resolution 512 \
    --max_resolution 1024 \
    --bucket_step 64 \
    --use_focal_loss \
    --use_class_balanced \
    --learning_rate 0.001 \
    --attention_heads 8
```

### 3. 运行推理

```bash
# 单张图像推理
python infer_full.py \
    --image_path path/to/image.jpg \
    --vae_checkpoint models/vae/pytorch_model.bin \
    --decoder_checkpoint models/decoder/pytorch_model.bin \
    --tags_csv_path your_tags.csv

# 批量推理
python infer_batch.sh /path/to/images/ models/ output/
```

## 📁 数据集格式

VAE-Tagger 需要两个核心文件来定义数据集：

### 1. 图像元数据文件 (JSON 格式)

这是一个包含图像信息和标签的 JSON 文件。**注意**: 本项目提供的 `create_test_dataset.py` 只是创建测试数据集的一种方法，你可以手动创建符合以下格式的数据文件：

```json
{
    "path/to/image1.jpg": "cat:1.0, outdoor:0.9, nature:0.8",
    "path/to/image2.png": "dog:1.0, indoor:0.95",
    "path/to/image3.jpg": "cat:1.0, dog:0.8, outdoor:0.7, large:1.0"
}
```

**格式说明:**
- **键**: 图像文件路径（相对或绝对路径）
- **值**: 用逗号分隔的标签字符串，包含置信度分数
- **标签格式**: `标签名:置信度分数`
- **置信度**: 0.0-1.0之间的浮点数（可选，省略时默认为1.0）

### 2. 标签定义文件 (CSV 格式)

定义所有可能标签的 CSV 文件：

```csv
name,count
cat,150
dog,120
outdoor,200
indoor,180
nature,90
large,75
medium,85
small,60
```

**字段说明:**
- `name`: 标签名称 (必须与 JSON 文件中的标签匹配)
- `count`: 该标签在数据集中出现的次数 (用于统计)

### 3. 数据集创建指南

#### 手动创建数据集的步骤:

1. **准备图像文件**: 将所有图像放在一个或多个目录中
2. **创建标签定义**: 列出所有可能的标签并分配ID
3. **标注图像**: 为每张图像分配相应的标签
4. **生成JSON文件**: 按照上述格式创建元数据文件

#### 示例目录结构(其实无所谓，csv指向所有出现的tag，json指向所有图片路径就行):
```
dataset/
├── images/
│   ├── cats/
│   │   ├── cat001.jpg
│   │   └── cat002.png
│   ├── dogs/
│   │   ├── dog001.jpg
│   │   └── dog002.jpg
│   └── mixed/
│       └── animal001.jpg
├── metadata.json
└── tags.csv
```

## 🎯 训练模式

VAE-Tagger 支持多种训练策略以适应不同使用场景：

### 1. 简化模式 (默认)
```bash
python train_full.py --use_simplified_loss
```
- **适用于**: 快速原型开发，资源有限的环境
- **特性**: 简化的损失计算，更快的训练速度
- **损失组成**: 分类损失 + 三元组学习

### 2. 标准模式
```bash
python train_full.py --use_focal_loss --use_class_balanced
```
- **适用于**: 生产环境训练，不平衡数据集
- **特性**: 高级损失函数，稳健的优化
- **损失组成**: Focal Loss + 类平衡 + 语义学习

### 3. 研究模式
```bash
python train_full.py --use_adaptive_weights --use_focal_loss
```
- **适用于**: 实验研究，追求最佳性能
- **特性**: 自适应优化，完整的VAE训练
- **损失组成**: 所有损失 + 自动权重平衡

## 📊 损失函数指南

### 可用损失函数

| 损失类型 | 使用场景 | 命令标志 |
|---------|---------|---------|
| **Focal Loss** | 不平衡数据集 | `--use_focal_loss` |
| **类平衡损失** | 长尾分布 | `--use_class_balanced` |
| **三元组损失** | 语义相似性 | 默认启用 |
| **对比损失** | 三元组损失的替代 | 修改配置 |
| **自适应权重** | 自动优化 | `--use_adaptive_weights` |

## 🏗️ 模型架构

### 核心组件
- **FLUX VAE 编码器**: 预训练扩散模型编码器，提供稳健的图像特征
- **语义解码器**: 基于多层注意力机制的分类器  
- **三元组学习**: 用于语义相似性的对比学习
- **自适应损失**: 自平衡优化

### 架构流程
```
图像 → FLUX VAE → 潜在特征 → 注意力解码器 → 标签预测
  ↓                ↓
重构 ← 语义学习 ← 分类损失
```

## 📚 API 参考

### 核心训练脚本

#### `train_full.py`
完整的端到端训练流水线。

**关键参数:**
- `--use_simplified_loss`: 启用简化训练模式
- `--use_focal_loss`: 对不平衡数据应用Focal Loss
- `--use_adaptive_weights`: 启用自动损失平衡
- `--similarity_type`: 选择'cosine'或'euclidean'相似度

#### `train_vae.py`
带三元组学习的VAE专用训练。

**关键参数:**
- `--use_simplified_vae_loss`: 简化VAE训练(仅KL监控)
- `--kl_weight`: KL散度损失权重(默认: 0.01)
- `--triplet_weight`: 三元组损失权重(默认: 1.0)

#### `train_decoder.py`
分类解码器训练。

**关键参数:**
- `--use_focal_loss`: 启用Focal Loss
- `--use_class_balanced`: 启用类平衡损失
- `--focal_alpha`, `--focal_gamma`: Focal Loss参数

### 实用工具脚本

- `validate_data.py`: 数据集验证和统计
- `evaluation.py`: 全面的模型评估
- `analyze_resolutions.py`: 数据集分辨率分析
- `vae_reconstruction_test.py`: VAE重构可视化

### 推理脚本

- `infer_full.py`: 单图像推理
- `infer_vae.py`: 仅VAE推理
- `infer_batch.sh`: 批处理脚本

## 🔧 配置

### 模型配置 (`vae_config.json`)
```json
{
  "sample_size": 512,
  "in_channels": 3,
  "out_channels": 3,
  "latent_channels": 16,
  "use_quant_conv": false,
  "scaling_factor": 0.3611,
  "shift_factor": 0.1159
}
```

### 训练配置
关键参数可通过命令行参数或配置文件调整：

- **学习率**: `--learning_rate` (默认: 1e-4)
- **批大小**: `--train_batch_size` (默认: 4)
- **损失权重**: `--reconstruction_weight`, `--kl_weight` 等
- **优化设置**: `--lr_scheduler_type`, `--max_grad_norm`
