import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    """对比损失"""
    def __init__(self, margin=1.0, similarity_type='cosine'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.similarity_type = similarity_type
    
    def forward(self, embedding1, embedding2, labels1, labels2):
        if self.similarity_type == 'cosine':
            # 余弦距离
            emb1_norm = F.normalize(embedding1, p=2, dim=1)
            emb2_norm = F.normalize(embedding2, p=2, dim=1)
            distance = 1 - (emb1_norm * emb2_norm).sum(dim=1)
        else:
            # 欧几里得距离
            distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Jaccard相似度
        label_overlap = (labels1 * labels2).sum(dim=1)
        label_union = (labels1 + labels2 - labels1 * labels2).sum(dim=1)
        label_similarity = label_overlap / (label_union + 1e-8)
        
        # 标签相似度对比损失
        similar_mask = label_similarity > 0.3  # 可调整阈值
        dissimilar_mask = ~similar_mask
        
        # 相似样本距离应该小
        similar_loss = similar_mask.float() * distance ** 2
        dissimilar_loss = dissimilar_mask.float() * torch.clamp(self.margin - distance, min=0.0) ** 2
        weight = torch.where(similar_mask, label_similarity, 1 - label_similarity)
        
        return ((similar_loss + dissimilar_loss) * weight).mean()

class FocalLoss(nn.Module):
    """Focal Loss处理类别不平衡问题"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    """类别平衡损失"""
    def __init__(self, beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, inputs, targets, samples_per_class):
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        weights = torch.tensor(weights, dtype=torch.float32, device=inputs.device)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_loss = bce_loss * weights.unsqueeze(0)
        return weighted_loss.mean()

class ImprovedTripletLoss(nn.Module):
    """三元组损失"""
    def __init__(self, margin=1.0, similarity_type='cosine'):
        super(ImprovedTripletLoss, self).__init__()
        self.margin = margin
        self.similarity_type = similarity_type
    
    def forward(self, anchor, positive, negative, anchor_labels=None, positive_labels=None):
        if self.similarity_type == 'cosine':
            # 余弦相似度
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            positive_norm = F.normalize(positive, p=2, dim=1)
            negative_norm = F.normalize(negative, p=2, dim=1)
            
            pos_dist = 1 - (anchor_norm * positive_norm).sum(dim=1)
            neg_dist = 1 - (anchor_norm * negative_norm).sum(dim=1)
        else:
            # 欧几里得距离
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # 基本三元组损失
        basic_loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # 如果提供了标签，使用标签重叠度调整损失权重
        if anchor_labels is not None and positive_labels is not None:
            label_overlap = (anchor_labels * positive_labels).sum(dim=1)
            # 重叠度越高，损失权重越大（因为这样的正样本更重要）
            weight = 1.0 + 0.5 * (label_overlap / (anchor_labels.sum(dim=1) + 1e-8))
            basic_loss = basic_loss * weight
        
        return basic_loss.mean()

class AdaptiveLossWeights(nn.Module):
    """损失权重调整器"""
    def __init__(self, num_losses=4, temperature=1.0):
        super(AdaptiveLossWeights, self).__init__()
        self.num_losses = num_losses
        self.temperature = temperature
        # 可学习的权重参数
        self.log_weights = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        # 计算归一化权重
        weights = F.softmax(self.log_weights / self.temperature, dim=0)
        # 计算加权损失
        weighted_loss = sum(w * loss for w, loss in zip(weights, losses))
        
        return weighted_loss, weights

class SimplifiedCombinedLoss(nn.Module):
    """简化组合损失函数，支持三元组损失或对比损失"""
    def __init__(self, 
                 classification_weight=1.0,
                 triplet_weight=0.5,
                 contrastive_weight=0.0,
                 use_focal_loss=True,
                 use_class_balanced=False,
                 use_contrastive=False,
                 focal_alpha=1.0,
                 focal_gamma=2.0,
                 triplet_margin=1.0,
                 contrastive_margin=1.0,  # 对比损失margin
                 similarity_type='cosine'):
        super(SimplifiedCombinedLoss, self).__init__()
        self.classification_weight = classification_weight
        self.triplet_weight = triplet_weight
        self.contrastive_weight = contrastive_weight
        self.use_contrastive = use_contrastive
        
        # 分类损失
        if use_focal_loss:
            self.classification_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()
        
        self.use_class_balanced = use_class_balanced
        if use_class_balanced:
            self.class_balanced_loss_fn = ClassBalancedLoss()
        
        # 语义学习损失：三元组损失或对比损失
        if use_contrastive:
            self.contrastive_loss_fn = ContrastiveLoss(margin=contrastive_margin, similarity_type=similarity_type)
        else:
            self.triplet_loss_fn = ImprovedTripletLoss(margin=triplet_margin, similarity_type=similarity_type)
    
    def forward(self, 
                z_a, 
                z_p, 
                z_n=None,
                classification_logits=None,
                classification_targets=None,
                anchor_labels=None,
                positive_labels=None,
                negative_labels=None,
                samples_per_class=None):
        loss_dict = {}
        total_loss = 0
        
        # 语义学习损失
        if self.use_contrastive and self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss_fn(
                z_a.reshape(z_a.size(0), -1),
                z_p.reshape(z_p.size(0), -1),
                anchor_labels,
                positive_labels
            )
            total_loss += self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss
        elif self.triplet_weight > 0:
            # 三元组损失：anchor, positive, negative
            triplet_loss = self.triplet_loss_fn(
                z_a.reshape(z_a.size(0), -1),
                z_p.reshape(z_p.size(0), -1),
                z_n.reshape(z_n.size(0), -1),
                anchor_labels,
                positive_labels
            )
            total_loss += self.triplet_weight * triplet_loss
            loss_dict['triplet_loss'] = triplet_loss
        
        # 分类损失
        if classification_logits is not None and classification_targets is not None:
            if self.use_class_balanced and samples_per_class is not None:
                classification_loss = self.class_balanced_loss_fn(
                    classification_logits, classification_targets, samples_per_class
                )
            else:
                classification_loss = self.classification_loss_fn(classification_logits, classification_targets)
            
            total_loss += self.classification_weight * classification_loss
            loss_dict['classification_loss'] = classification_loss
        
        # 组合损失
        loss_dict['total_loss'] = total_loss
        
        # 权重信息
        if self.use_contrastive:
            loss_dict['weights'] = torch.tensor([
                self.contrastive_weight,
                self.classification_weight
            ])
        else:
            loss_dict['weights'] = torch.tensor([
                self.triplet_weight,
                self.classification_weight
            ])
        
        return loss_dict

class CombinedLoss(nn.Module):
    """完整版组合损失"""
    def __init__(self, 
                 reconstruction_weight=0.01,
                 kl_weight=1e-2,  # log变换后的KL损失
                 triplet_weight=1.0,
                 classification_weight=1.0,
                 use_focal_loss=True,
                 use_class_balanced=False,
                 use_adaptive_weights=False,
                 focal_alpha=1.0,
                 focal_gamma=2.0,
                 triplet_margin=1.0,
                 similarity_type='cosine'):
        super(CombinedLoss, self).__init__()
        
        # 基础权重
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.triplet_weight = triplet_weight
        self.classification_weight = classification_weight
        
        if use_focal_loss:
            self.classification_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()
        
        self.use_class_balanced = use_class_balanced
        if use_class_balanced:
            self.class_balanced_loss_fn = ClassBalancedLoss()
        
        self.triplet_loss_fn = ImprovedTripletLoss(margin=triplet_margin, similarity_type=similarity_type)
        
        # 自适应权重
        self.use_adaptive_weights = use_adaptive_weights
        if use_adaptive_weights:
            self.adaptive_weights = AdaptiveLossWeights(num_losses=4)
    
    def forward(self, 
                reconstruction, 
                target_images, 
                posterior_a,
                posterior_p, 
                posterior_n,
                z_a, 
                z_p, 
                z_n,
                classification_logits,
                classification_targets,
                anchor_labels=None,
                positive_labels=None,
                samples_per_class=None):
        # 重建损失
        reconstruction_loss = F.mse_loss(reconstruction, target_images)
        
        # KL损失，log稳定了一下
        kl_raw = (posterior_a.kl() + posterior_p.kl() + posterior_n.kl()) / 3
        if len(kl_raw.shape) > 1:
            kl_mean = kl_raw.mean()  # 空间维度平均
        else:
            kl_mean = kl_raw.mean()
        
        # 使用log变换稳定KL损失
        kl_loss = torch.log(1 + kl_mean / 10000)
        
        # 三元组损失
        triplet_loss = self.triplet_loss_fn(
            z_a.reshape(z_a.size(0), -1),
            z_p.reshape(z_p.size(0), -1),
            z_n.reshape(z_n.size(0), -1),
            anchor_labels,
            positive_labels
        )
        
        # 分类损失
        if self.use_class_balanced and samples_per_class is not None:
            classification_loss = self.class_balanced_loss_fn(
                classification_logits, classification_targets, samples_per_class
            )
        else:
            classification_loss = self.classification_loss_fn(classification_logits, classification_targets)
        
        # 组合损失
        losses = [reconstruction_loss, kl_loss, triplet_loss, classification_loss]
        
        if self.use_adaptive_weights:
            total_loss, adaptive_weights = self.adaptive_weights(losses)
            loss_dict = {
                'total_loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'triplet_loss': triplet_loss,
                'classification_loss': classification_loss,
                'adaptive_weights': adaptive_weights
            }
        else:
            total_loss = (self.reconstruction_weight * reconstruction_loss +
                         self.kl_weight * kl_loss +
                         self.triplet_weight * triplet_loss +
                         self.classification_weight * classification_loss)
            
            loss_dict = {
                'total_loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'triplet_loss': triplet_loss,
                'classification_loss': classification_loss,
                'weights': torch.tensor([
                    self.reconstruction_weight,
                    self.kl_weight,
                    self.triplet_weight,
                    self.classification_weight
                ])
            }
        
        return loss_dict

def compute_class_distribution(dataset):
    class_counts = np.zeros(len(dataset.tags))
    for image_path in dataset.image_paths:
        labels = dataset.image_labels[image_path]
        for i, label_val in enumerate(labels):
            if label_val > 0:
                class_counts[i] += 1
    return class_counts

if __name__ == "__main__":
    batch_size, num_classes, feature_dim = 4, 10, 512
    reconstruction = torch.randn(batch_size, 3, 256, 256)
    target_images = torch.randn(batch_size, 3, 256, 256)
    class MockPosterior:
        def __init__(self, mean, logvar):
            self.mean = mean
            self.logvar = logvar
        def kl(self):
            return -0.5 * torch.sum(1 + self.logvar - self.mean.pow(2) - self.logvar.exp(), dim=1)
    
    posterior_a = MockPosterior(torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim))
    posterior_p = MockPosterior(torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim))
    posterior_n = MockPosterior(torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim))
    
    z_a = torch.randn(batch_size, feature_dim)
    z_p = torch.randn(batch_size, feature_dim)
    z_n = torch.randn(batch_size, feature_dim)
    
    classification_logits = torch.randn(batch_size, num_classes)
    classification_targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print("\n简化版损失函数")
    simplified_loss_fn = SimplifiedCombinedLoss(use_focal_loss=True)
    
    simplified_loss_dict = simplified_loss_fn(
        z_a, z_p, z_n,
        classification_logits, classification_targets,
        classification_targets, classification_targets
    )
    
    print("简化版损失函数结果:")
    for key, value in simplified_loss_dict.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"{key}: {value.item():.4f}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    print("\n完整版损失函数")
    loss_fn = CombinedLoss(use_focal_loss=True, use_adaptive_weights=True)
    
    loss_dict = loss_fn(
        reconstruction, target_images,
        posterior_a, posterior_p, posterior_n,
        z_a, z_p, z_n,
        classification_logits, classification_targets,
        classification_targets, classification_targets
    )
    
    print("完整版损失函数结果:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"{key}: {value.item():.4f}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
