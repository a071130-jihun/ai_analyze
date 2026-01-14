#!/usr/bin/env python3
"""
Best model checkpoint 평가 스크립트
Usage: 
  CNN:  python evaluate_checkpoint.py --model ./output/sleep_stage_cnn.pt --stages 3
  CRNN: python evaluate_checkpoint.py --model ./output/sleep_stage_crnn.pt --stages 3 --seq_len 11
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import pickle
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, 
    classification_report, confusion_matrix
)

from sleep_stage_classifier.models.classifier import get_model
from sleep_stage_classifier.models.sequence_model import SleepSequenceModel

STAGE_NAMES_5 = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_NAMES_3 = {0: "Wake", 1: "NREM", 2: "REM"}


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


def convert_to_3stage(labels):
    mapping = np.array([0, 1, 1, 1, 2], dtype=np.int32)
    return mapping[np.clip(labels, 0, 4)]


def remap_labels(labels):
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    mapping_arr = np.zeros(max(unique_labels) + 1, dtype=np.int32)
    for old, new in label_map.items():
        mapping_arr[old] = new
    remapped = mapping_arr[labels]
    return remapped, label_map


def standardize_features(features, robust=True):
    if robust:
        p5 = np.percentile(features, 5)
        p95 = np.percentile(features, 95)
        mean = (p5 + p95) / 2
        std = (p95 - p5) / 2 + 1e-8
    else:
        mean = features.mean()
        std = features.std() + 1e-8
    return (features - mean) / std


def evaluate_batch(model, features, device="cuda", batch_size=256):
    model.eval()
    n_samples = len(features)
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = features[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            del batch_tensor, outputs
            if device == "cuda":
                torch.cuda.empty_cache()
    
    return np.array(all_preds), np.array(all_probs)


def evaluate_sequence(model, features, labels, seq_len, device="cuda", batch_size=64):
    if device == "cuda":
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    model.eval()
    half_seq = seq_len // 2
    
    valid_indices = list(range(half_seq, len(labels) - half_seq))
    all_preds = []
    all_probs = []
    valid_labels = []
    
    print(f"  Evaluating {len(valid_indices)} samples with seq_len={seq_len}...")
    
    with torch.no_grad():
        for batch_start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[batch_start:batch_start + batch_size]
            
            batch_seqs = []
            batch_labels = []
            for center_idx in batch_indices:
                start_idx = center_idx - half_seq
                end_idx = center_idx + half_seq + 1
                seq = torch.FloatTensor(features[start_idx:end_idx]).unsqueeze(1)
                batch_seqs.append(seq)
                batch_labels.append(labels[center_idx])
            
            batch_tensor = torch.stack(batch_seqs).to(device)
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            valid_labels.extend(batch_labels)
            
            del batch_tensor, outputs
            if device == "cuda":
                torch.cuda.empty_cache()
            
            if (batch_start // batch_size) % 100 == 0:
                print(f"    Progress: {batch_start}/{len(valid_indices)}")
    
    return np.array(all_preds), np.array(all_probs), np.array(valid_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model checkpoint")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--cache", default="./cache", help="Cache directory")
    parser.add_argument("--stages", type=int, default=3, choices=[3, 5], help="Number of stages")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio")
    parser.add_argument("--seq_len", type=int, default=0, help="Sequence length (0=auto detect, >0 for sequence model)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    stage_names = STAGE_NAMES_3 if args.stages == 3 else STAGE_NAMES_5
    num_classes = 3 if args.stages == 3 else 5
    
    print(f"\n[1/4] Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch'] + 1}")
        if "metric_value" in checkpoint:
            print(f"  Best {checkpoint.get('metric_name', 'metric')}: {checkpoint['metric_value']:.4f}")
    else:
        state_dict = checkpoint
    
    is_sequence_model = any("backbone" in k or "temporal_lstm" in k for k in state_dict.keys())
    
    if is_sequence_model:
        seq_len = args.seq_len if args.seq_len > 0 else 11
        print(f"  Detected: Sequence Model (seq_len={seq_len})")
        model = SleepSequenceModel(num_classes=num_classes, hidden_dim=256, seq_len=seq_len)
    else:
        seq_len = 0
        print(f"  Detected: CNN Model")
        model = get_model(model_type="cnn", num_classes=num_classes)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  Model loaded!")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    
    print(f"\n[2/4] Loading data from cache: {args.cache}")
    features, labels = load_from_cache(args.cache)
    print(f"  Total samples: {len(labels)}")
    
    if args.stages == 3:
        print(f"\n  Converting to 3-stage (Wake/NREM/REM)...")
        labels = convert_to_3stage(labels)
    
    labels, label_map = remap_labels(labels)
    print(f"  Label mapping: {label_map}")
    print(f"  Classes: {len(label_map)}")
    
    print(f"\n[3/4] Splitting data (seed=42, test_ratio={args.test_ratio})...")
    np.random.seed(42)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * args.test_ratio)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    print(f"  Test samples: {len(test_labels)}")
    
    print(f"\n  Standardizing features (robust)...")
    test_features_norm = standardize_features(test_features, robust=True)
    
    print(f"\n[4/4] Evaluating...")
    
    if is_sequence_model:
        predictions, probabilities, test_labels = evaluate_sequence(
            model, test_features_norm, test_labels, 
            seq_len=seq_len, device=device, batch_size=args.batch_size
        )
    else:
        predictions, probabilities = evaluate_batch(
            model, test_features_norm, 
            device=device, batch_size=args.batch_size
        )
    
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(test_labels, predictions)
    
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS (TEST SET)")
    print("=" * 70)
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
    target_names = [stage_names.get(i, f"Class_{i}") for i in present_classes]
    
    print("\n" + "-" * 70)
    print("  Classification Report:")
    print("-" * 70)
    print(classification_report(
        test_labels, predictions,
        labels=present_classes,
        target_names=target_names,
        zero_division=0
    ))
    
    print("-" * 70)
    print("  Confusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(test_labels, predictions, labels=present_classes)
    
    header = "        " + "  ".join([f"{name:>6}" for name in target_names])
    print(header)
    print("        " + "-" * (len(target_names) * 8))
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:>6}" for v in row])
        print(f"{target_names[i]:>6} | {row_str}")
    
    print("\n" + "-" * 70)
    print("  Class Distribution:")
    print("-" * 70)
    unique, counts = np.unique(test_labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = stage_names.get(u, f"Class_{u}")
        pred_count = np.sum(predictions == u)
        recall = np.sum((predictions == u) & (test_labels == u)) / c if c > 0 else 0
        print(f"    {name}: {c} actual, {pred_count} predicted (recall: {recall*100:.1f}%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
