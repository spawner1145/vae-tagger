import json
import argparse
from pathlib import Path
import subprocess

def run_inference_on_images(vae_checkpoint, vae_config_path, decoder_checkpoint, 
                           tags_csv_path, image_dir, output_base_dir, resolution=256, 
                           confidence_threshold=0.3, max_images=10):
    image_paths = list(Path(image_dir).glob("*.jpg"))[:max_images]
    results = {}
    
    print(f"开始对 {len(image_paths)} 张图像进行推理测试...")
    
    for i, img_path in enumerate(image_paths):
        output_dir = Path(output_base_dir) / f"result_{i:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "D:/Python311/python.exe", "infer_full.py",
            "--vae_checkpoint", vae_checkpoint,
            "--vae_config_path", vae_config_path,
            "--decoder_checkpoint", decoder_checkpoint,
            "--tags_csv_path", tags_csv_path,
            "--image_path", str(img_path),
            "--output_dir", str(output_dir),
            "--resolution", str(resolution),
            "--confidence_threshold", str(confidence_threshold)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                result_file = output_dir / "classification_results.json"
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        inference_result = json.load(f)
                    results[str(img_path)] = inference_result[list(inference_result.keys())[0]]
                    print(f"{img_path.name}: 成功")
                else:
                    print(f"{img_path.name}: 结果文件不存在")
            else:
                print(f"{img_path.name}: 推理失败 - {result.stderr}")
                
        except Exception as e:
            print(f"{img_path.name}: 执行错误 - {e}")
    
    return results

def load_ground_truth(data_json_path):
    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ground_truth = {}
    for img_path, label_str in data.items():
        tags = []
        for part in label_str.split(', '):
            tag_name = part.split(':')[0].strip()
            tags.append(tag_name)
        normalized_path = str(Path(img_path).as_posix())
        ground_truth[normalized_path] = tags
    
    return ground_truth

def calculate_metrics(predictions, ground_truth):
    total_images = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    exact_matches = 0
    
    detailed_results = []
    
    for img_path, pred_data in predictions.items():
        normalized_path = str(Path(img_path).as_posix())
        true_tags = None
        for gt_path, gt_tags in ground_truth.items():
            if Path(gt_path).name == Path(normalized_path).name:
                true_tags = gt_tags
                break
        
        if true_tags is None:
            print(f"警告: 找不到 {img_path} 的真实标签")
            continue
        pred_tags = [item['tag'] for item in pred_data['predicted_tags']]
        true_set = set(true_tags)
        pred_set = set(pred_tags)
        
        intersection = true_set & pred_set
        
        if len(pred_set) > 0:
            precision = len(intersection) / len(pred_set)
        else:
            precision = 0
            
        if len(true_set) > 0:
            recall = len(intersection) / len(true_set)
        else:
            recall = 1  # 没有真实标签时认为召回率为1
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        exact_match = 1 if true_set == pred_set else 0
        
        detailed_results.append({
            'image': Path(img_path).name,
            'true_tags': true_tags,
            'pred_tags': pred_tags,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': exact_match
        })
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        exact_matches += exact_match
        total_images += 1
    
    if total_images > 0:
        avg_precision = total_precision / total_images
        avg_recall = total_recall / total_images
        avg_f1 = total_f1 / total_images
        exact_match_rate = exact_matches / total_images
    else:
        avg_precision = avg_recall = avg_f1 = exact_match_rate = 0
    
    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'exact_match_rate': exact_match_rate,
        'total_images': total_images,
        'detailed_results': detailed_results
    }

def main():
    parser = argparse.ArgumentParser(description="批量推理测试")
    parser.add_argument("--vae_checkpoint", type=str, default="full_output/best_vae/diffusion_pytorch_model.safetensors")
    parser.add_argument("--vae_config_path", type=str, default="full_output/best_vae/config.json")
    parser.add_argument("--decoder_checkpoint", type=str, default="full_output/best_decoder/pytorch_model.bin")
    parser.add_argument("--tags_csv_path", type=str, default="test_dataset/tags.csv")
    parser.add_argument("--image_dir", type=str, default="test_dataset/images")
    parser.add_argument("--data_json_path", type=str, default="test_dataset/data.json")
    parser.add_argument("--output_dir", type=str, default="batch_inference_results")
    parser.add_argument("--max_images", type=int, default=10, help="测试的最大图像数量")
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    print("批量推理测试开始")
    predictions = run_inference_on_images(
        args.vae_checkpoint, args.vae_config_path, args.decoder_checkpoint,
        args.tags_csv_path, args.image_dir, args.output_dir,
        args.resolution, args.confidence_threshold, args.max_images
    )
    ground_truth = load_ground_truth(args.data_json_path)
    metrics = calculate_metrics(predictions, ground_truth)
    print("\n整体性能指标")
    print(f"平均精确率: {metrics['avg_precision']:.4f}")
    print(f"平均召回率: {metrics['avg_recall']:.4f}")
    print(f"平均F1分数: {metrics['avg_f1']:.4f}")
    print(f"完全匹配率: {metrics['exact_match_rate']:.4f}")
    print(f"测试图像数: {metrics['total_images']}")
    print("\n详细结果")
    for result in metrics['detailed_results']:
        print(f"{result['image']}:")
        print(f"  真实标签: {result['true_tags']}")
        print(f"  预测标签: {result['pred_tags']}")
        print(f"  精确率: {result['precision']:.3f}, 召回率: {result['recall']:.3f}, F1: {result['f1']:.3f}")
        print()
    output_file = Path(args.output_dir) / "batch_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()