import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd


def validate_dataset(json_path: str, tags_csv_path: str, output_dir: str = "data_validation", fix: bool = False):
    json_path = Path(json_path)
    tags_csv_path = Path(tags_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise FileNotFoundError(f"找不到数据JSON: {json_path}")
    if not tags_csv_path.exists():
        raise FileNotFoundError(f"找不到标签CSV: {tags_csv_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tags_df = pd.read_csv(tags_csv_path)
    if 'name' not in tags_df.columns:
        raise ValueError("标签CSV缺少 'name' 列")

    valid_tags = set(tags_df['name'].astype(str).tolist())

    total = len(data)
    missing_images = []
    unreadable = []
    images_with_unknown_tags = {}
    empty_label_images = []
    tag_counter = Counter()

    for i, (img_path, tag_str) in enumerate(data.items()):
        p = Path(img_path)
        if not p.exists():
            missing_images.append(img_path)
            continue

        # parse tags
        tags = []
        tag_str = (tag_str or "").strip()
        if tag_str:
            for chunk in tag_str.split(','):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if ':' in chunk:
                    name, _ = chunk.split(':', 1)
                    name = name.strip()
                else:
                    name = chunk
                tags.append(name)

        if not tags:
            empty_label_images.append(img_path)
        else:
            unknown = [t for t in tags if t not in valid_tags]
            if unknown:
                images_with_unknown_tags[img_path] = unknown
            for t in tags:
                if t in valid_tags:
                    tag_counter[t] += 1

        if (i + 1) % 100 == 0:
            print(f"已检查 {i + 1}/{total}")

    report = {
        "total_images": total,
        "existing_images": total - len(missing_images),
        "missing_images": len(missing_images),
        "empty_label_images": len(empty_label_images),
        "images_with_unknown_tags": len(images_with_unknown_tags),
        "top_tags": tag_counter.most_common(50),
    }

    # 保存详细报告
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (output_dir / "missing_images.json").write_text(
        json.dumps(missing_images, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (output_dir / "empty_label_images.json").write_text(
        json.dumps(empty_label_images, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (output_dir / "unknown_tags_by_image.json").write_text(
        json.dumps(images_with_unknown_tags, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    print("数据校验完成：")
    print(f"  总图像数: {report['total_images']}")
    print(f"  存在的图像: {report['existing_images']}")
    print(f"  丢失的图像: {report['missing_images']}")
    print(f"  无标签的图像: {report['empty_label_images']}")
    print(f"  含未知标签的图像: {report['images_with_unknown_tags']}")
    print(f"  报告保存到: {output_dir}")

    if fix:
        fixed = {}
        for img_path, tag_str in data.items():
            if img_path in missing_images:
                continue
            # 过滤未知标签
            kept_parts = []
            for chunk in (tag_str or "").split(','):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if ':' in chunk:
                    name, score = chunk.split(':', 1)
                    name = name.strip()
                    score = score.strip()
                else:
                    name = chunk
                    score = "1.0"
                if name in valid_tags:
                    kept_parts.append(f"{name}:{score}")
            if kept_parts:
                fixed[img_path] = ", ".join(kept_parts)

        fixed_path = output_dir / "data.cleaned.json"
        fixed_path.write_text(json.dumps(fixed, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"已输出清洗后的数据: {fixed_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="校验并统计数据集JSON/CSV的一致性与完整性")
    parser.add_argument("--json_path", type=str, required=True, help="数据集 JSON 文件路径")
    parser.add_argument("--tags_csv_path", type=str, required=True, help="标签 CSV（需包含'name'列）")
    parser.add_argument("--output_dir", type=str, default="data_validation", help="报告输出目录")
    parser.add_argument("--fix", action="store_true", help="输出清洗后的 data.cleaned.json（移除缺失图像、未知标签）")
    args = parser.parse_args()

    validate_dataset(args.json_path, args.tags_csv_path, args.output_dir, args.fix)
