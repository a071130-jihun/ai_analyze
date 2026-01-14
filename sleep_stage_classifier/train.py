from typing import Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LambdaLR
)

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)

from .config import TrainConfig, SLEEP_STAGE_NAMES


def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr: float, base_lr: float):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return max(min_lr / base_lr, cosine_decay)
    return LambdaLR(optimizer, lr_lambda)


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
        class_weights: np.ndarray = None,
        use_amp: bool = True,
        use_compile: bool = True,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        self.config = train_config or TrainConfig()
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model = model.to(self.device)
        
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("  torch.compile() enabled")
            except Exception as e:
                print(f"  torch.compile() failed, using eager mode: {e}")
        
        self.use_focal_loss = use_focal_loss
        
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"  Using FocalLoss (gamma={focal_gamma})")
        else:
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
        
        self.best_val_metric = -float('inf') if self.config.best_model_metric != "loss" else float('inf')
        self.best_model_path = None
        
        if self.config.save_best_model:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "learning_rate": []
        }
    
    def _create_scheduler(self, num_epochs: int, steps_per_epoch: int = None):
        scheduler_type = self.config.scheduler_type.lower()
        
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=self.config.min_lr
            )
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                min_lr=self.config.min_lr,
                verbose=True
            )
        elif scheduler_type == "onecycle":
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch required for OneCycleLR")
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_type == "warmup_cosine":
            return get_warmup_cosine_scheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=num_epochs,
                min_lr=self.config.min_lr,
                base_lr=self.config.learning_rate
            )
        else:
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=self.config.min_lr
            )
    
    def _get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
    
    def _is_better(self, current_metric: float) -> bool:
        metric_name = self.config.best_model_metric
        if metric_name == "loss":
            return current_metric < self.best_val_metric
        else:
            return current_metric > self.best_val_metric
    
    def _save_best_model(self, epoch: int, metric_value: float):
        if not self.config.save_best_model:
            return
        
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
        
        self.best_model_path = os.path.join(
            self.config.checkpoint_dir,
            f"best_model_epoch{epoch+1}_{self.config.best_model_metric}{metric_value:.4f}.pt"
        )
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_name": self.config.best_model_metric,
            "metric_value": metric_value,
            "history": self.history,
            "config": self.config
        }, self.best_model_path)
        
        print(f"  ★ Best model saved: {self.best_model_path}")
    
    def train_epoch(self, train_loader: DataLoader, use_mixup: bool = True, mixup_alpha: float = 0.2) -> Tuple[float, float]:
        from .augmentation import mixup_data, mixup_criterion
        
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if use_mixup and np.random.random() < 0.5:
                    mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, mixup_alpha)
                    outputs = self.model(mixed_x)
                    loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
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
        
        self.scheduler = self._create_scheduler(
            num_epochs=num_epochs,
            steps_per_epoch=len(train_loader)
        )
        
        is_plateau = self.config.scheduler_type.lower() == "plateau"
        is_onecycle = self.config.scheduler_type.lower() == "onecycle"
        
        best_model_state = None
        
        print(f"  Scheduler: {self.config.scheduler_type}")
        print(f"  Best model metric: {self.config.best_model_metric}")
        
        for epoch in range(num_epochs):
            current_lr = self._get_current_lr()
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            if is_plateau:
                self.scheduler.step(val_metrics["loss"])
            elif not is_onecycle:
                self.scheduler.step()
            
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["learning_rate"].append(current_lr)
            
            print(f"Epoch {epoch+1}/{num_epochs} [LR: {current_lr:.2e}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            metric_map = {
                "f1": val_metrics["f1"],
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"]
            }
            current_metric = metric_map.get(self.config.best_model_metric, val_metrics["f1"])
            
            if self._is_better(current_metric):
                self.best_val_metric = current_metric
                best_model_state = self.model.state_dict().copy()
                self._save_best_model(epoch, current_metric)
            
            if self.early_stopping(val_metrics["loss"]):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n  Loaded best model (best {self.config.best_model_metric}: {self.best_val_metric:.4f})")
        
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


def compute_class_weights(labels: np.ndarray, power: float = 0.35, max_weight: float = 4.0) -> np.ndarray:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    freq = counts / total
    weights = (1.0 / freq) ** power
    weights = weights / weights.min()
    weights = np.clip(weights, 1.0, max_weight)
    
    full_weights = np.ones(max(unique) + 1)
    for u, w in zip(unique, weights):
        full_weights[u] = w
    
    print(f"  Class weights: {dict(zip(unique, weights))}")
    return full_weights
