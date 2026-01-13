#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from sleep_stage_classifier.models.classifier import get_model

STAGE_NAMES = ["Wake", "N1", "N2", "REM"]


def load_model(model_path: str, num_classes: int = 4):
    model = get_model(model_type="cnn", num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("history", {})


def evaluate():
    model_path = "./output/sleep_stage_cnn.pt"
    data_path = "./output/evaluation_data.npz"
    
    print("=" * 60)
    print("Model Evaluation (Test Set Only)")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"\nError: {data_path} not found.")
        print("Run training first: python run_pipeline.py --mode train")
        return
    
    print("\n[1] Loading saved data splits...")
    data = np.load(data_path)
    features = data["features"]
    labels = data["labels"]
    test_indices = data["test_indices"]
    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    
    print(f"  Total samples: {len(labels)}")
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    print(f"\n  Using TEST SET only: {len(test_labels)} samples")
    
    print("\n[2] Loading model...")
    num_classes = len(np.unique(labels))
    model, history = load_model(model_path, num_classes=num_classes)
    print("  Model loaded!")
    
    print("\n[3] Running predictions on TEST SET...")
    features_tensor = torch.FloatTensor(test_features).unsqueeze(1)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        predictions = outputs.argmax(dim=1).numpy()
        probabilities = torch.softmax(outputs, dim=1).numpy()
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS (TEST SET)")
    print("=" * 60)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(test_labels, predictions)
    
    print(f"\n  Accuracy:        {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  F1 (macro):      {f1_macro:.4f}")
    print(f"  F1 (weighted):   {f1_weighted:.4f}")
    print(f"  Cohen's Kappa:   {kappa:.4f}")
    
    print("\n  Per-class metrics:")
    present_classes = np.unique(np.concatenate([test_labels, predictions]))
    target_names = [STAGE_NAMES[i] for i in present_classes]
    report = classification_report(
        test_labels, predictions, 
        labels=present_classes,
        target_names=target_names, 
        zero_division=0
    )
    print(report)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    cm = confusion_matrix(test_labels, predictions, labels=present_classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=axes[0, 0], cmap='Blues', colorbar=False)
    axes[0, 0].set_title('Confusion Matrix - TEST SET (Counts)')
    
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    disp2 = ConfusionMatrixDisplay(cm_normalized, display_labels=target_names)
    disp2.plot(ax=axes[0, 1], cmap='Blues', colorbar=False, values_format='.2f')
    axes[0, 1].set_title('Confusion Matrix - TEST SET (Normalized)')
    
    time_epochs = np.arange(len(test_labels))
    axes[1, 0].plot(time_epochs, test_labels, 'b-', label='Ground Truth', alpha=0.7, linewidth=1)
    axes[1, 0].plot(time_epochs, predictions, 'r--', label='Predicted', alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('Test Sample Index')
    axes[1, 0].set_ylabel('Sleep Stage')
    axes[1, 0].set_yticks(present_classes)
    axes[1, 0].set_yticklabels(target_names)
    axes[1, 0].legend()
    axes[1, 0].set_title('Test Set: Ground Truth vs Predicted')
    axes[1, 0].grid(True, alpha=0.3)
    
    if history:
        epochs_trained = range(1, len(history.get('train_loss', [])) + 1)
        if 'train_loss' in history:
            axes[1, 1].plot(epochs_trained, history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in history:
            axes[1, 1].plot(epochs_trained, history['val_loss'], 'r-', label='Val Loss')
        if 'val_f1' in history:
            ax2 = axes[1, 1].twinx()
            ax2.plot(epochs_trained, history['val_f1'], 'g--', label='Val F1')
            ax2.set_ylabel('F1 Score', color='g')
            ax2.legend(loc='upper right')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend(loc='upper left')
        axes[1, 1].set_title('Training History')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No training history', ha='center', va='center')
        axes[1, 1].set_title('Training History')
    
    plt.tight_layout()
    
    save_path = "./evaluation_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
  This evaluation uses ONLY the held-out TEST SET
  (data not seen during training or validation)

  Cohen's Kappa interpretation:
    - < 0.20: Poor
    - 0.21-0.40: Fair  
    - 0.41-0.60: Moderate
    - 0.61-0.80: Substantial
    - > 0.80: Almost perfect
""")


if __name__ == "__main__":
    evaluate()
