"""
contrastive_loss.py — Noise Contrastive Estimation (NCE) Loss
Implements the contrastive learning objective from the paper.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class NCELoss(nn.Module):
    """
    Noise Contrastive Estimation Loss.
    
    Simple intuition:
    - Given an anchor image (e.g., a happy face),
    - A positive sample (another happy face) should be CLOSE in feature space
    - Negative samples (angry, sad, etc.) should be FAR in feature space
    
    Temperature T controls how sharp the distribution is:
    - Low T (0.07): Very sharp — model must be very sure
    - High T (1.0): Softer — more tolerance for uncertainty
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: L2-normalized feature vectors, shape [batch, dim]
            labels: class labels, shape [batch]
        
        Returns:
            loss: scalar contrastive loss value
        """
        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix: [batch, batch]
                # Each entry (i,j) is cosine similarity between features i and j
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        batch_size = features.shape[0]
        
        # Create mask: 1 where labels match (positive pairs), 0 otherwise
        # We exclude the diagonal (each sample with itself)
        labels = labels.unsqueeze(1)  # [batch, 1]
        mask = (labels == labels.T).float()  # [batch, batch]
        
        # Remove self-similarity from mask
        mask.fill_diagonal_(0)
        
        # For numerical stability: subtract max before softmax
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()
        
        # Compute log-softmax over similarities
        # exp_sim: exponential of similarity scores
        exp_sim = torch.exp(sim_matrix)
        
        # Exclude self from denominator
        self_mask = torch.eye(batch_size, device=features.device).bool()
        exp_sim = exp_sim.masked_fill(self_mask, 0)
        
        # Sum of all negative (and positive) exponentials
        sum_exp = exp_sim.sum(dim=1, keepdim=True)
        
        # Log probability of positive pairs
        log_prob = sim_matrix - torch.log(sum_exp + 1e-8)
        
        # Average over positive pairs
        # Count positive pairs per sample
        num_positives = mask.sum(dim=1)
        
        # Avoid division by zero
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=features.device)
        
        loss = -(mask * log_prob).sum(dim=1)
        loss = loss[valid] / num_positives[valid]
        
        return loss.mean()
class CombinedLoss(nn.Module):
    """
    Combines Cross-Entropy (classification) + NCE (contrastive) losses.
    
    Total Loss = alpha * CE_Loss + (1-alpha) * NCE_Loss
    """
    
    def __init__(self, num_classes, temperature=0.5, alpha=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.nce_loss = NCELoss(temperature)
        self.alpha = alpha
    
    def forward(self, logits, contrastive_features, labels):
        """
        Args:
            logits: raw class scores from classifier, shape [batch, num_classes]
            contrastive_features: projected features for NCE, shape [batch, 128]
            labels: ground truth emotion labels, shape [batch]
        
        Returns:
            total_loss, ce_loss_val, nce_loss_val
        """
        ce = self.ce_loss(logits, labels)
        nce = self.nce_loss(contrastive_features, labels)
        
        total = self.alpha * ce + (1 - self.alpha) * nce
        return total, ce.item(), nce.item()
