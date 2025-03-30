#!/usr/bin/env python3
"""
Knowledge distillation utilities for transferring knowledge from larger
teacher models to smaller student models.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelDistillation")

class DistillationLoss(nn.Module):
    """
    Distillation loss combining cross-entropy and KL divergence loss
    for transferring knowledge from teacher to student.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize the distillation loss.
        
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for balancing distillation and task loss
            reduction: Reduction method for the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate the distillation loss.
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            targets: Ground truth labels (optional)
            
        Returns:
            Combined loss value
        """
        # Distillation loss - soft targets from teacher
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction=self.reduction
        ) * (self.temperature ** 2)
        
        # If targets are provided, calculate task-specific loss
        if targets is not None:
            task_loss = self.ce_loss(student_logits, targets)
            # Combine the losses
            return self.alpha * task_loss + (1 - self.alpha) * distillation_loss
        
        return distillation_loss


class DetectionDistillationLoss(nn.Module):
    """
    Distillation loss for object detection models, combining
    classification, box regression, and feature-level distillation.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha_class: float = 0.5,
        alpha_bbox: float = 0.5,
        alpha_feature: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize the detection distillation loss.
        
        Args:
            temperature: Temperature for softening probability distributions
            alpha_class: Weight for classification distillation loss
            alpha_bbox: Weight for bounding box regression distillation loss
            alpha_feature: Weight for feature-level distillation loss
            reduction: Reduction method for the loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha_class = alpha_class
        self.alpha_bbox = alpha_bbox
        self.alpha_feature = alpha_feature
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the detection distillation loss.
        
        Args:
            student_outputs: Dictionary of outputs from student model
            teacher_outputs: Dictionary of outputs from teacher model
            targets: Ground truth targets (optional)
            
        Returns:
            Dictionary with individual and combined loss values
        """
        losses = {}
        
        # Classification logits distillation
        if "logits" in student_outputs and "logits" in teacher_outputs:
            losses["class_distill"] = F.kl_div(
                F.log_softmax(student_outputs["logits"] / self.temperature, dim=1),
                F.softmax(teacher_outputs["logits"] / self.temperature, dim=1),
                reduction=self.reduction
            ) * (self.temperature ** 2) * self.alpha_class
        
        # Bounding box regression distillation
        if "boxes" in student_outputs and "boxes" in teacher_outputs:
            losses["bbox_distill"] = self.mse_loss(
                student_outputs["boxes"],
                teacher_outputs["boxes"]
            ) * self.alpha_bbox
        
        # Feature-level distillation
        if "features" in student_outputs and "features" in teacher_outputs:
            # Handle different feature map sizes with adaptive pooling
            student_feats = student_outputs["features"]
            teacher_feats = teacher_outputs["features"]
            
            # Ensure features are the same size
            if student_feats.shape != teacher_feats.shape:
                # Adjust student features to match teacher
                student_feats = F.adaptive_avg_pool2d(
                    student_feats,
                    output_size=teacher_feats.shape[2:]
                )
            
            losses["feature_distill"] = self.mse_loss(
                student_feats,
                teacher_feats
            ) * self.alpha_feature
        
        # Combine all distillation losses
        losses["total_distill"] = sum(
            loss for name, loss in losses.items() if name != "total_distill"
        )
        
        # If targets are provided, calculate task-specific loss
        # This would depend on the specific detection model being used
        
        return losses


class ModelDistiller:
    """
    Framework for knowledge distillation from a teacher model to a student model.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = "auto"
    ):
        """
        Initialize the model distiller.
        
        Args:
            teacher_model: Teacher model (larger, more accurate)
            student_model: Student model (smaller, to be trained)
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        self.device = self._resolve_device(device)
        
        # Set up teacher model
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model.eval()  # Teacher is fixed during distillation
        
        # Set up student model
        self.student_model = student_model.to(self.device)
        self.student_model.train()
        
        # Set up output directory
        self.output_dir = Path("distilled_models")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Distiller initialized with device: {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 0.001,
        temperature: float = 4.0,
        alpha: float = 0.5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_best: bool = True,
        early_stopping: int = 0,
        model_name: str = "distilled_model",
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the student model with knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            lr: Learning rate
            temperature: Temperature for distillation loss
            alpha: Weight balancing distillation and task loss
            optimizer: Optimizer to use (defaults to Adam)
            loss_fn: Loss function to use (defaults to DistillationLoss)
            scheduler: Learning rate scheduler (optional)
            save_best: Whether to save the best model
            early_stopping: Number of epochs for early stopping (0 to disable)
            model_name: Name for saving the model
            callback: Optional callback function to execute after each epoch
            
        Returns:
            Dictionary with training and validation metrics
        """
        logger.info(f"Starting distillation training for {epochs} epochs")
        
        # Set up optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        
        # Set up loss function
        if loss_fn is None:
            loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)
        
        # Training metrics
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        # Best model tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.student_model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - teacher
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)
                
                # Forward pass - student
                student_outputs = self.student_model(inputs)
                
                # Compute loss
                loss = loss_fn(student_outputs, teacher_outputs, targets)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Batch {batch_idx}/{len(train_loader)} - "
                              f"Loss: {loss.item():.4f}")
            
            # Calculate epoch metrics
            train_loss = train_loss / train_samples
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate(val_loader, loss_fn)
                history["val_loss"].append(val_loss)
                
                # Update learning rate scheduler if provided
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_model(model_name + "_best.pth")
                    logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping > 0 and patience_counter >= early_stopping:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"Time: {time.time() - epoch_start_time:.2f}s")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Time: {time.time() - epoch_start_time:.2f}s")
            
            # Execute callback if provided
            if callback is not None:
                metrics = {"epoch": epoch, "train_loss": train_loss}
                if val_loader is not None:
                    metrics["val_loss"] = val_loss
                callback(epoch, metrics)
        
        # Save final model
        self._save_model(model_name + "_final.pth")
        logger.info("Distillation training completed")
        
        return history
    
    def _validate(self, val_loader: DataLoader, loss_fn: nn.Module) -> float:
        """
        Validate the student model.
        
        Args:
            val_loader: DataLoader for validation data
            loss_fn: Loss function to use
            
        Returns:
            Validation loss
        """
        self.student_model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass - teacher
                teacher_outputs = self.teacher_model(inputs)
                
                # Forward pass - student
                student_outputs = self.student_model(inputs)
                
                # Compute loss
                loss = loss_fn(student_outputs, teacher_outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
        
        return val_loss / val_samples
    
    def _save_model(self, filename: str) -> str:
        """
        Save the student model.
        
        Args:
            filename: Name of the file to save the model to
            
        Returns:
            Path to the saved model
        """
        output_path = self.output_dir / filename
        torch.save(self.student_model.state_dict(), output_path)
        return str(output_path)
    
    def evaluate(
        self,
        test_loader: DataLoader,
        metrics_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the student model.
        
        Args:
            test_loader: DataLoader for test data
            metrics_fn: Function to compute metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.student_model.eval()
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.student_model(inputs)
                
                # Store outputs and targets
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Concatenate outputs and targets
        if isinstance(all_outputs[0], torch.Tensor):
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics if provided
        if metrics_fn is not None:
            return metrics_fn(all_outputs, all_targets)
        
        # Default metrics for classification
        if isinstance(all_outputs, torch.Tensor) and all_outputs.dim() > 1:
            preds = torch.argmax(all_outputs, dim=1)
            accuracy = (preds == all_targets).float().mean().item()
            return {"accuracy": accuracy}
        
        return {}


# Example usage
def create_distilled_model(
    teacher_path: str,
    student_path: str,
    dataset_path: str,
    output_dir: str = "distilled_models",
    epochs: int = 10,
    batch_size: int = 8,
    temperature: float = 4.0,
    alpha: float = 0.5,
    lr: float = 0.001,
    device: str = "auto"
) -> str:
    """
    Create a distilled model from a teacher model.
    
    Args:
        teacher_path: Path to the teacher model
        student_path: Path to the student model
        dataset_path: Path to the dataset for distillation
        output_dir: Directory to save the distilled model
        epochs: Number of training epochs
        batch_size: Batch size for training
        temperature: Temperature for distillation
        alpha: Weight for balancing distillation and task loss
        lr: Learning rate
        device: Device to use
        
    Returns:
        Path to the distilled model
    """
    try:
        # This is a placeholder - in a real implementation, you would need
        # to load the actual models and dataset based on the provided paths
        from model_optimizer import ModelOptimizer
        
        # Load teacher model
        teacher_optimizer = ModelOptimizer(model_path=teacher_path, device=device)
        teacher_model = teacher_optimizer.model
        
        # Load student model
        student_optimizer = ModelOptimizer(model_path=student_path, device=device)
        student_model = student_optimizer.model
        
        # Create data loaders
        # This would depend on the specific dataset being used
        logger.info(f"Loading dataset from {dataset_path}")
        train_loader = None  # Replace with actual data loader
        val_loader = None  # Replace with actual data loader
        
        # Set up distiller
        distiller = ModelDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            device=device
        )
        
        # Set output directory
        distiller.output_dir = Path(output_dir)
        distiller.output_dir.mkdir(exist_ok=True)
        
        # Train with distillation
        distiller.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            temperature=temperature,
            alpha=alpha,
            save_best=True,
            model_name="distilled_model"
        )
        
        # Return path to best model
        return str(distiller.output_dir / "distilled_model_best.pth")
        
    except Exception as e:
        logger.error(f"Error creating distilled model: {e}")
        raise


def main():
    """Command-line interface for model distillation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model distillation utility")
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher model")
    parser.add_argument("--student", type=str, required=True, help="Path to student model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default="distilled_models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for balancing losses")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Perform distillation
    output_path = create_distilled_model(
        teacher_path=args.teacher,
        student_path=args.student,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
        lr=args.lr
    )
    
    logger.info(f"Distilled model saved to {output_path}")


if __name__ == "__main__":
    main() 