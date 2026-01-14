#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

from sleep_stage_classifier.models.classifier import get_model

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def evaluate_batch(model, features, device="cuda", batch_size=256):
    model.eval()
    n_samples = len(features)
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = features[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
            outputs = model(batch_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            del batch_tensor, outputs
            torch.cuda.empty_cache()
    
    return np.array(all_preds)


def main():
    model_path = "./output/sleep_stage_cnn.pt"
    data_path = "./output/evaluation_data.npz"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    num_classes = 5
    model = get_model(model_type="cnn", num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    
    test_features = data["features"][data["test_indices"]]
    test_labels = data["test_labels"]
    
    print(f"Test samples: {len(test_labels)}")
    
    print("Evaluating...")
    predictions = evaluate_batch(model, test_features, device=device)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(test_labels, predictions)
    
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy:      {accuracy*100:.1f}%")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    
    present_classes = sorted(set(test_labels) | set(predictions))
    target_names = [STAGE_NAMES.get(i, f"Class_{i}") for i in present_classes]
    
    print("\n  Classification Report:")
    print(classification_report(
        test_labels, predictions,
        labels=present_classes,
        target_names=target_names,
        zero_division=0
    ))
    print("=" * 60)


if __name__ == "__main__":
    main()
