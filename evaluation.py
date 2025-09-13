import torch
import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, 
    f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class MultiLabelEvaluator:
    def __init__(self, class_names=None, device="cuda"):
        self.class_names = class_names
        self.device = device
        self.reset_metrics()
    
    def reset_metrics(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions, targets, probabilities=None):
        """
        Args:
            predictions: (batch_size, num_classes) 
            targets: (batch_size, num_classes) 
            probabilities: (batch_size, num_classes) 
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        self.all_predictions.append(predictions)
        self.all_targets.append(targets)
        if probabilities is not None:
            self.all_probabilities.append(probabilities)
    
    def compute_metrics(self, threshold=0.5):
        if not self.all_targets:
            raise ValueError("update")
        
        y_true = np.vstack(self.all_targets)
        y_pred = np.vstack(self.all_predictions)
        
        if self.all_probabilities:
            y_prob = np.vstack(self.all_probabilities)
        else:
            y_prob = y_pred
        
        metrics = {}
        metrics['accuracy'] = self._compute_subset_accuracy(y_true, y_pred)
        metrics['hamming_loss'] = self._compute_hamming_loss(y_true, y_pred)

        for average in ['micro', 'macro', 'weighted']:
            metrics[f'precision_{average}'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            metrics[f'recall_{average}'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            metrics[f'f1_{average}'] = f1_score(y_true, y_pred, average=average, zero_division=0)

        try:
            metrics['mAP'] = average_precision_score(y_true, y_prob, average='macro')
            metrics['mAP_micro'] = average_precision_score(y_true, y_prob, average='micro')
            metrics['mAP_weighted'] = average_precision_score(y_true, y_prob, average='weighted')
        except ValueError as e:
            print(f"mAP: {e}")
            metrics['mAP'] = 0.0
            metrics['mAP_micro'] = 0.0
            metrics['mAP_weighted'] = 0.0
        
        per_class_metrics = self._compute_per_class_metrics(y_true, y_pred, y_prob)
        metrics['per_class'] = per_class_metrics
        
        return metrics
    
    def _compute_subset_accuracy(self, y_true, y_pred):
        return (y_true == y_pred).all(axis=1).mean()
    
    def _compute_hamming_loss(self, y_true, y_pred):
        return (y_true != y_pred).mean()
    
    def _compute_per_class_metrics(self, y_true, y_pred, y_prob):
        num_classes = y_true.shape[1]
        per_class = {}
        
        for i in range(num_classes):
            class_name = self.class_names[i] if self.class_names else f"Class_{i}"
             
            if y_true[:, i].sum() == 0:
                per_class[class_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'ap': 0.0,
                    'support': 0
                }
            elif y_true[:, i].sum() == len(y_true):
                per_class[class_name] = {
                    'precision': (y_pred[:, i] == 1).mean(),
                    'recall': 1.0,
                    'f1': 2 * (y_pred[:, i] == 1).mean() / (1 + (y_pred[:, i] == 1).mean()) if (y_pred[:, i] == 1).sum() > 0 else 0.0,
                    'ap': 1.0,
                    'support': int(y_true[:, i].sum())
                }
            else:
                try:
                    precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                    recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                    f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                    ap = average_precision_score(y_true[:, i], y_prob[:, i])
                    
                    per_class[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'ap': ap,
                        'support': int(y_true[:, i].sum())
                    }
                except Exception as e:
                    print(f" {class_name} : {e}")
                    per_class[class_name] = {
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'ap': 0.0,
                        'support': int(y_true[:, i].sum())
                    }
        
        return per_class
    
    def print_metrics(self, metrics, detailed=True):
        print(f"    (Subset Accuracy): {metrics['accuracy']:.4f}")
        print(f"    (Hamming Loss):   {metrics['hamming_loss']:.4f}")
        
        for metric_type in ['precision', 'recall', 'f1']:
            print(f"   {metric_type.capitalize()}:")
            for avg_type in ['micro', 'macro', 'weighted']:
                value = metrics[f'{metric_type}_{avg_type}']
                print(f"     {avg_type}: {value:.4f}")
        
        print(f"\n mAP (mean Average Precision):")
        print(f"   Macro:    {metrics['mAP']:.4f}")
        print(f"   Micro:    {metrics['mAP_micro']:.4f}")
        print(f"   Weighted: {metrics['mAP_weighted']:.4f}")

        if detailed and 'per_class' in metrics:
            print(f"{'':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AP':<10} {'Support':<10}")
            
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"{class_name:<20} "
                      f"{class_metrics['precision']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} "
                      f"{class_metrics['f1']:<10.4f} "
                      f"{class_metrics['ap']:<10.4f} "
                      f"{class_metrics['support']:<10}")
    
    def save_metrics(self, metrics, output_path):
        overall_metrics = {k: v for k, v in metrics.items() if k != 'per_class'}
        
        import json
        with open(output_path.replace('.csv', '_overall.json'), 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
        
        if 'per_class' in metrics:
            df = pd.DataFrame(metrics['per_class']).T
            df.index.name = 'class_name'
            df.to_csv(output_path)
            print(f" : {output_path}")

def evaluate_model(model, decoder, test_loader, class_names, device="cuda", threshold=0.5, output_dir=None):
    model.eval()
    decoder.eval()
    
    evaluator = MultiLabelEvaluator(class_names, device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="")):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            latent_vectors = model.encode(pixel_values)

            logits = decoder(latent_vectors)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()

            evaluator.update(predictions, labels, probabilities)

    metrics = evaluator.compute_metrics(threshold)

    evaluator.print_metrics(metrics)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "evaluation_results.csv")
        evaluator.save_metrics(metrics, output_path)
    
    return metrics

def find_optimal_threshold(model, decoder, val_loader, class_names, device="cuda", output_dir=None):
    model.eval()
    decoder.eval()

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=""):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            latent_vectors = model.encode(pixel_values)
            logits = decoder(latent_vectors)
            probabilities = torch.sigmoid(logits)
            
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    y_prob = np.vstack(all_probabilities)
    y_true = np.vstack(all_targets)
    optimal_thresholds = {}
    thresholds = np.arange(0.1, 0.9, 0.05)
    for i, class_name in enumerate(class_names):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_prob[:, i] > threshold).astype(int)
            y_true_int = y_true[:, i].astype(int)  # 确保是整数类型
            if y_true_int.sum() > 0:  # 确保有正样本
                f1 = f1_score(y_true_int, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        optimal_thresholds[class_name] = {
            'threshold': best_threshold,
            'f1_score': best_f1
        }
    
    # macro F1
    best_global_f1 = 0
    best_global_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_global = (y_prob > threshold).astype(int)
        y_true_int = y_true.astype(int)  # 确保标签也是整数类型
        f1_macro = f1_score(y_true_int, y_pred_global, average='macro', zero_division=0)
        if f1_macro > best_global_f1:
            best_global_f1 = f1_macro
            best_global_threshold = threshold
    
    results = {
        'global_threshold': best_global_threshold,
        'global_f1': best_global_f1,
        'per_class_thresholds': optimal_thresholds
    }

    print(f"Global Threshold: {best_global_threshold:.3f} (Macro F1: {best_global_f1:.4f})")
    print("\nPer-Class Thresholds:")
    for class_name, info in optimal_thresholds.items():
        print(f"  {class_name:<20}: {info['threshold']:.3f} (F1: {info['f1_score']:.4f})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "optimal_thresholds.json")
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"{output_path}")
    
    return results