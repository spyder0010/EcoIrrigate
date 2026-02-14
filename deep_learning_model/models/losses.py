"""
Custom Loss Functions for Precision Agriculture Deep Learning
===============================================================

Implements specialized loss functions:
1. Weighted MSE for multi-horizon forecasting
2. Huber loss for robust calibration
3. Multi-task loss with task balancing
4. Uncertainty-aware losses

Author: Research Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for multi-horizon forecasting.
    
    Assigns higher weights to near-term predictions (more important).
    """
    
    def __init__(self, horizon_weights=None):
        """
        Initialize weighted MSE loss.
        
        Args:
            horizon_weights: List of weights for each horizon
                           Default: [1.0, 0.8, 0.6, 0.4] for [1h, 6h, 12h, 24h]
        """
        super(WeightedMSELoss, self).__init__()
        
        if horizon_weights is None:
            horizon_weights = [1.0, 0.8, 0.6, 0.4]
        
        self.horizon_weights = torch.FloatTensor(horizon_weights)
        
    def forward(self, predictions, targets):
        """
        Compute weighted MSE.
        
        Args:
            predictions: [batch, num_horizons]
            targets: [batch, num_horizons]
            
        Returns:
            Weighted MSE loss
        """
        # Move weights to same device
        if self.horizon_weights.device != predictions.device:
            self.horizon_weights = self.horizon_weights.to(predictions.device)
        
        # Squared errors per horizon
        squared_errors = (predictions - targets) ** 2  # [batch, num_horizons]
        
        # Weight and sum
        weighted_errors = squared_errors * self.horizon_weights.unsqueeze(0)
        
        # Mean over batch and horizons
        loss = weighted_errors.mean()
        
        return loss


class HuberLoss(nn.Module):
    """
    Huber loss for robust regression (less sensitive to outliers than MSE).
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold for switching from quadratic to linear
        """
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, predictions, targets):
        """
        Compute Huber loss.
        
        Args:
            predictions: [batch, 1] or [batch]
            targets: [batch, 1] or [batch]
            
        Returns:
            Huber loss
        """
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # Quadratic for small errors, linear for large errors
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        
        loss = torch.where(abs_error <= self.delta, quadratic, linear)
        
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with automatic task weighting.
    
    Uses uncertainty-based weighting (Kendall et al., 2018):
    L_total = task1_loss / (2 * σ1^2) + task2_loss / (2 * σ2^2) + log(σ1 * σ2)
    
    Learns optimal task weights during training.
    """
    
    def __init__(self, num_tasks=2):
        """
        Initialize multi-task loss.
        
        Args:
            num_tasks: Number of tasks (2 for calibration + forecasting)
        """
        super(MultiTaskLoss, self).__init__()
        
        # Learnable log variance parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, task_losses):
        """
        Compute multi-task loss.
        
        Args:
            task_losses: List of losses for each task
            
        Returns:
            Combined multi-task loss
        """
        total_loss = 0
        
        for i, loss in enumerate(task_losses):
            # Precision (inverse variance)
            precision = torch.exp(-self.log_vars[i])
            
            # Weighted loss + regularization
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss
    
    def get_task_weights(self):
        """Get current task weights for logging."""
        return torch.exp(-self.log_vars).detach().cpu().numpy()


class FocalMSELoss(nn.Module):
    """
    Focal MSE loss - focuses on hard-to-predict examples.
    
    Inspired by Focal Loss for classification, adapted for regression.
    """
    
    def __init__(self, gamma=2.0, alpha=1.0):
        """
        Initialize Focal MSE loss.
        
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Scaling factor
        """
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        """
        Compute Focal MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            
        Returns:
            Focal MSE loss
        """
        mse = (predictions - targets) ** 2
        
        # Focal weight: larger for larger errors
        focal_weight = (mse + 1e-8) ** (self.gamma / 2)
        
        loss = self.alpha * focal_weight * mse
        
        return loss.mean()


class QuantileLoss(nn.Module):
    """
    Quantile loss for uncertainty quantification.
    
    Predicts prediction intervals (e.g., 5th, 50th, 95th percentiles).
    """
    
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        """
        Initialize quantile loss.
        
        Args:
            quantiles: List of quantiles to predict
        """
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions, targets):
        """
        Compute quantile loss.
        
        Args:
            predictions: [batch, num_quantiles]
            targets: [batch, 1] (ground truth)
            
        Returns:
            Quantile loss
        """
        total_loss = 0
        
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i:i+1]
            
            # Asymmetric loss (penalize under/over-prediction differently)
            loss = torch.max((q - 1) * errors, q * errors)
            total_loss += loss.mean()
        
        return total_loss / len(self.quantiles)


def main():
    """Test loss functions."""
    
    # Test Weighted MSE
    print("### TESTING WEIGHTED MSE LOSS ###")
    loss_fn = WeightedMSELoss(horizon_weights=[1.0, 0.8, 0.6, 0.4])
    pred = torch.randn(32, 4)
    target = torch.randn(32, 4)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test Huber Loss
    print("\n### TESTING HUBER LOSS ###")
    loss_fn = HuberLoss(delta=1.0)
    pred = torch.randn(32, 1)
    target = torch.randn(32, 1)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test Multi-Task Loss
    print("\n### TESTING MULTI-TASK LOSS ###")
    loss_fn = MultiTaskLoss(num_tasks=2)
    task1_loss = torch.tensor(0.5)
    task2_loss = torch.tensor(1.2)
    total_loss = loss_fn([task1_loss, task2_loss])
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Task weights: {loss_fn.get_task_weights()}")
    
    # Test Focal MSE
    print("\n### TESTING FOCAL MSE LOSS ###")
    loss_fn = FocalMSELoss(gamma=2.0)
    pred = torch.randn(32, 1)
    target = torch.randn(32, 1)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test Quantile Loss
    print("\n### TESTING QUANTILE LOSS ###")
    loss_fn = QuantileLoss(quantiles=[0.05, 0.5, 0.95])
    pred = torch.randn(32, 3)  # 3 quantiles
    target = torch.randn(32, 1)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
