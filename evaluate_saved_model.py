#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

from sleep_stage_classifier.models.classifier import get_model

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def load_from_cache(cache_dir="./cache"):
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*_features.pkl")))
    
    all_features = []
    all_labels = []
    
    for cache_file in cache_files:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        all_features.append(data['features'])
        all_labels.append(data['labels'])
        print(f"  Loaded: {os.path.basename(cache_file)} ({len(data['labels'])} samples)")
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels


def remap_labels(labels):
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped = np.array([label_map[l] for l in labels])
    return remapped, label_map


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
            if device == "cuda":
                torch.cuda.empty_cache()
    
    return np.array(all_preds)


def main():
    model_path = "./output/sleep_stage_cnn.pt"
    cache_dir = "./cache"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    num_classes = 5
    model = get_model(model_type="cnn", num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("  Model loaded!")
    
    print("\nLoading data from cache...")
    features, labels = load_from_cache(cache_dir)
    print(f"\n  Total samples: {len(labels)}")
    
    labels, label_map = remap_labels(labels)
    print(f"  Classes: {len(label_map)}")
    
    print("\nSplitting data (same seed as training)...")
    np.random.seed(42)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * 0.2)
    test_indices = indices[:test_size]
    
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    print(f"  Test samples: {len(test_labels)}")
    
    print("\nEvaluating...")
    predictions = evaluate_batch(model, test_features, device=device)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(test_labels, predictions)
    
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (TEST SET)")
    print("=" * 60)
    print(f"\n  Accuracy:      {accuracy*100:.1f}%")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    
    if kappa < 0.20:
        kappa_level = "Poor"
    elif kappa < 0.40:
        kappa_level = "Fair"
    elif kappa < 0.60:
        kappa_level = "Moderate"
    elif kappa < 0.80:
        kappa_level = "Substantial"
    else:
        kappa_level = "Almost Perfect"
    print(f"  Kappa Level:   {kappa_level}")
    
    present_classes = sorted(set(test_labels) | set(predictions))
    target_names = [STAGE_NAMES.get(i, f"Class_{i}") for i in present_classes]
    
    print("\n  Classification Report:")
    print(classification_report(
        test_labels, predictions,
        labels=present_classes,
        target_names=target_names,
        zero_division=0
    ))
    
    print("  Class Distribution (Test Set):")
    unique, counts = np.unique(test_labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = STAGE_NAMES.get(u, f"Class_{u}")
        print(f"    {name}: {c} ({100*c/len(test_labels):.1f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
