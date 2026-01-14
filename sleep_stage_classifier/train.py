from typing import Dict, Tuple, Optional
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)

from .config import TrainConfig, SLEEP_STAGE_NAMES


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig = None,
        class_weights: np.ndarray = None
    ):
        self.config = train_config or TrainConfig()
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing
        )
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = None
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": []
        }
    
    def train_epoch(self, train_loader: DataLoader, use_mixup: bool = True, mixup_alpha: float = 0.2) -> Tuple[float, float]:
        from .augmentation import mixup_data, mixup_criterion
        
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if use_mixup and np.random.random() < 0.5:
                mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, mixup_alpha)
                outputs = self.model(mixed_x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
            else:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ) -> Dict:
        num_epochs = num_epochs or self.config.num_epochs
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        best_val_f1 = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step()
            
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_state = self.model.state_dict().copy()
            
            if self.early_stopping(val_metrics["loss"]):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def get_detailed_metrics(self, test_loader: DataLoader) -> Dict:
        metrics = self.evaluate(test_loader)
        
        unique_labels = np.unique(np.concatenate([metrics["labels"], metrics["predictions"]]))
        target_names = [SLEEP_STAGE_NAMES.get(i, f"Class_{i}") for i in unique_labels]
        
        report = classification_report(
            metrics["labels"],
            metrics["predictions"],
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(metrics["labels"], metrics["predictions"])
        
        return {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1"],
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": metrics["predictions"],
            "labels": metrics["labels"]
        }
    
    def save_model(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)


def compute_class_weights(labels: np.ndarray, power: float = 0.5) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    freq = counts / total
    weights = (1.0 / freq) ** power
    weights = weights / weights.min()
    
    full_weights = np.ones(max(unique) + 1)
    for u, w in zip(unique, weights):
        full_weights[u] = w
    
    print(f"  Class weights: {dict(zip(unique, weights))}")
    return full_weights
