import json
import argparse
from PIL import Image
from pathlib import Path
from collections import defaultdict

def analyze_image_resolutions(json_path, output_dir="resolution_analysis"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    resolutions = []
    aspect_ratios = []
    resolution_counts = defaultdict(int)
    aspect_ratio_counts = defaultdict(int)
    for i, (image_path, _) in enumerate(data.items()):
        try:
            if not Path(image_path).exists():
                print(f"图像不存在: {image_path}")
                continue
                
            with Image.open(image_path) as img:
                w, h = img.size
                resolutions.append((w, h))
                aspect_ratio = round(w / h, 2)
                aspect_ratios.append(aspect_ratio)
                resolution_counts[f"{w}x{h}"] += 1
                aspect_ratio_counts[aspect_ratio] += 1
                
                if (i + 1) % 100 == 0:
                    print(f"已分析 {i + 1}/{len(data)} 张图像")
                    
        except Exception as e:
            print(f"无法读取图像 {image_path}: {e}")
    
    print(f"\n基本统计:")
    print(f"总图像数量: {len(resolutions)}")
    print(f"不同分辨率数量: {len(resolution_counts)}")
    print(f"不同长宽比数量: {len(aspect_ratio_counts)}")
    
    # 分辨率统计
    print(f"\n最常见的分辨率 (前10个):")
    sorted_resolutions = sorted(resolution_counts.items(), key=lambda x: x[1], reverse=True)
    for resolution, count in sorted_resolutions[:10]:
        percentage = (count / len(resolutions)) * 100
        print(f"  {resolution}: {count} 张 ({percentage:.1f}%)")
    
    # 长宽比统计
    print(f"\n最常见的长宽比 (前10个):")
    sorted_aspects = sorted(aspect_ratio_counts.items(), key=lambda x: x[1], reverse=True)
    for aspect, count in sorted_aspects[:10]:
        percentage = (count / len(aspect_ratios)) * 100
        if aspect == 1.0:
            print(f"  1:1 (正方形): {count} 张 ({percentage:.1f}%)")
        elif aspect > 1:
            print(f"  {aspect}:1 (横向): {count} 张 ({percentage:.1f}%)")
        else:
            print(f"  1:{1/aspect:.2f} (纵向): {count} 张 ({percentage:.1f}%)")
    
    # 尺寸范围分析
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    
    print(f"\n尺寸范围分析:")
    print(f"宽度范围: {min(widths)} - {max(widths)} (平均: {sum(widths)//len(widths)})")
    print(f"高度范围: {min(heights)} - {max(heights)} (平均: {sum(heights)//len(heights)})")
    
    # 推荐处理策略
    print(f"\n处理策略建议:")
    
    square_ratio = aspect_ratio_counts.get(1.0, 0) / len(aspect_ratios)
    if square_ratio > 0.7:
        print("大部分图像为正方形，建议使用 'resize' 模式")
    elif square_ratio > 0.3:
        print("图像长宽比混合，建议使用 'center_crop' 模式")
    else:
        print("图像长宽比差异较大，建议:")
        print("   - 'center_crop': 保持主体内容，可能裁剪边缘")
        print("   - 'pad': 保持完整图像，添加黑边")
        print("   - 'resize_shorter': 缩放短边，然后裁剪")
    
    # 目标分辨率建议
    avg_area = sum(w*h for w, h in resolutions) / len(resolutions)
    suggested_resolution = int((avg_area ** 0.5) // 64 * 64)  # 向下取整到64的倍数
    
    print(f"\n建议训练分辨率:")
    print(f"基于平均图像面积: {suggested_resolution}x{suggested_resolution}")
    print(f"常用选择: 512x512 (快速训练) 或 1024x1024 (高质量)")
    
    return {
        'resolutions': resolutions,
        'aspect_ratios': aspect_ratios,
        'resolution_counts': resolution_counts,
        'aspect_ratio_counts': aspect_ratio_counts,
        'suggested_resolution': suggested_resolution
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析图像数据集的分辨率分布")
    parser.add_argument("--json_path", type=str, required=True, help="数据集JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="resolution_analysis", help="输出目录")
    
    args = parser.parse_args()
    
    try:
        results = analyze_image_resolutions(args.json_path, args.output_dir)
        print(f"\n分析完成！")
    except Exception as e:
        print(f"分析失败: {e}")
