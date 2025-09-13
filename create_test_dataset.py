import json
from PIL import Image, ImageDraw
import pandas as pd
from pathlib import Path
import random

def create_synthetic_dataset(output_dir="test_dataset", num_images=100):
    """创建合成数据集"""
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    shape_tags = ["circle", "square", "triangle", "rectangle"]
    color_tags = ["red", "blue", "green", "yellow", "purple"]
    size_tags = ["small", "medium", "large"]
    style_tags = ["solid", "outline", "gradient"]
    all_tags = shape_tags + color_tags + size_tags + style_tags
    data_dict = {}
    tag_counts = {tag: 0 for tag in all_tags}
    
    print(f"生成 {num_images} 张合成图像...")
    
    for i in range(num_images):
        shape = random.choice(shape_tags)
        color = random.choice(color_tags)
        size = random.choice(size_tags)
        style = random.choice(style_tags)
        selected_tags = [shape, color, size]
        if random.random() > 0.3:
            selected_tags.append(style)
        for tag in all_tags:
            if tag not in selected_tags and random.random() > 0.9:
                selected_tags.append(tag)
        for tag in selected_tags:
            tag_counts[tag] += 1
        img_size = 256
        img = Image.new('RGB', (img_size, img_size), 'white')
        draw = ImageDraw.Draw(img)
        color_map = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255), 
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128)
        }
        size_map = {
            'small': 30,
            'medium': 50,
            'large': 80
        }
        fill_color = color_map[color]
        shape_size = size_map[size]
        center = img_size // 2
        if shape == "circle":
            bbox = [center-shape_size, center-shape_size, 
                   center+shape_size, center+shape_size]
            if style == "solid":
                draw.ellipse(bbox, fill=fill_color)
            elif style == "outline":
                draw.ellipse(bbox, outline=fill_color, width=3)
            else:
                for r in range(shape_size, 0, -2):
                    alpha = int(255 * (r / shape_size))
                    grad_color = tuple(int(c * alpha / 255) for c in fill_color)
                    draw.ellipse([center-r, center-r, center+r, center+r], 
                               fill=grad_color)
        
        elif shape == "square":
            bbox = [center-shape_size, center-shape_size, 
                   center+shape_size, center+shape_size]
            if style == "solid":
                draw.rectangle(bbox, fill=fill_color)
            elif style == "outline":
                draw.rectangle(bbox, outline=fill_color, width=3)
            else:  # gradient
                for r in range(shape_size, 0, -2):
                    alpha = int(255 * (r / shape_size))
                    grad_color = tuple(int(c * alpha / 255) for c in fill_color)
                    draw.rectangle([center-r, center-r, center+r, center+r], 
                                 fill=grad_color)
        
        elif shape == "triangle":
            points = [(center, center-shape_size),
                     (center-shape_size, center+shape_size),
                     (center+shape_size, center+shape_size)]
            if style == "solid":
                draw.polygon(points, fill=fill_color)
            elif style == "outline":
                draw.polygon(points, outline=fill_color, width=3)
            else:  # gradient
                draw.polygon(points, fill=fill_color)
        
        elif shape == "rectangle":
            bbox = [center-shape_size, center-shape_size//2, 
                   center+shape_size, center+shape_size//2]
            if style == "solid":
                draw.rectangle(bbox, fill=fill_color)
            elif style == "outline":
                draw.rectangle(bbox, outline=fill_color, width=3)
            else:  # gradient
                draw.rectangle(bbox, fill=fill_color)

        img_filename = f"synthetic_{i:04d}.jpg"
        img_path = images_dir / img_filename
        img.save(img_path, quality=90)

        tag_strings = [f"{tag}:1.0" for tag in selected_tags]
        tag_string = ", ".join(tag_strings)
        relative_path = f"{output_dir}/images/{img_filename}"
        data_dict[relative_path] = tag_string
        
        if (i + 1) % 20 == 0:
            print(f"  已生成 {i + 1}/{num_images} 张图像")
    data_json_path = Path(output_dir) / "data.json"
    with open(data_json_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    tags_df = pd.DataFrame([
        {'name': tag, 'count': count} 
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        if count > 0
    ])
    tags_csv_path = Path(output_dir) / "tags.csv"
    tags_df.to_csv(tags_csv_path, index=False)
    print(f"输出目录: {output_dir}")
    print(f"图像数量: {num_images}")
    print(f"标签数量: {len([c for c in tag_counts.values() if c > 0])}")
    print(f"\n标签分布:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / num_images) * 100
            print(f"  {tag}: {count} 次 ({percentage:.1f}%)")
    
    return {
        'data_json': str(data_json_path),
        'tags_csv': str(tags_csv_path),
        'images_dir': str(images_dir),
        'num_images': num_images,
        'num_tags': len([c for c in tag_counts.values() if c > 0])
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="test_dataset")
    parser.add_argument("--num_images", type=int, default=100)
    args = parser.parse_args()
    
    create_synthetic_dataset(args.output_dir, args.num_images)
