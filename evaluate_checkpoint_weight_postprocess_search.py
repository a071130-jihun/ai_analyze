#!/usr/bin/env python3
"""
Grid search class weights + post-processing parameters.
Usage:
  python evaluate_checkpoint_weight_postprocess_search.py --model ./output/sleep_stage_cnn.pt --stages 3
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import pickle
import itertools
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from sleep_stage_classifier.models.classifier import get_model
from sleep_stage_classifier.models.sequence_model import (
    SleepSequenceModel,
    DeepSleepResNet,
    DeepSleepResNetLarge,
)

STAGE_NAMES_5 = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_NAMES_3 = {0: "Wake", 1: "NREM", 2: "REM"}


def load_from_cache(cache_dir="./cache"):
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*_features.pkl")))

    all_features = []
    all_labels = []

    for cache_file in cache_files:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        all_features.append(data["features"])
        all_labels.append(data["labels"])
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
            batch = features[i:i + batch_size]
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


def apply_class_weights(probs, weights):
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 1:
        raise ValueError("class_weights must be a 1D array")
    if probs.shape[1] != len(weights):
        raise ValueError(f"class_weights length {len(weights)} != num_classes {probs.shape[1]}")
    weighted = probs * weights[None, :]
    row_sums = weighted.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return weighted / row_sums


def apply_majority_vote(preds, window, num_classes):
    if window <= 1:
        return preds.copy()
    if window % 2 == 0:
        raise ValueError("window must be an odd number")
    pad = window // 2
    padded = np.pad(preds, (pad, pad), mode="edge")
    output = np.empty_like(preds)
    for i in range(len(preds)):
        window_vals = padded[i:i + window]
        counts = np.bincount(window_vals, minlength=num_classes)
        max_count = counts.max()
        candidates = np.flatnonzero(counts == max_count)
        if len(candidates) == 1:
            output[i] = candidates[0]
        else:
            center = preds[i]
            output[i] = center if center in candidates else candidates[0]
    return output


def apply_prob_average(probs, window):
    if window <= 1:
        return probs.copy()
    if window % 2 == 0:
        raise ValueError("window must be an odd number")
    pad = window // 2
    padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
    output = np.zeros_like(probs)
    for i in range(len(probs)):
        output[i] = padded[i:i + window].mean(axis=0)
    return output


def enforce_min_run(preds, min_len):
    if min_len <= 1:
        return preds.copy()
    output = preds.copy()
    runs = []
    start = 0
    for i in range(1, len(preds) + 1):
        if i == len(preds) or preds[i] != preds[start]:
            runs.append((start, i, preds[start]))
            start = i
    for idx, (start, end, label) in enumerate(runs):
        run_len = end - start
        if run_len >= min_len:
            continue
        left = runs[idx - 1] if idx - 1 >= 0 else None
        right = runs[idx + 1] if idx + 1 < len(runs) else None
        if left is None and right is None:
            continue
        if left is None:
            replacement = right[2]
        elif right is None:
            replacement = left[2]
        else:
            if left[2] == right[2]:
                replacement = left[2]
            else:
                left_len = left[1] - left[0]
                right_len = right[1] - right[0]
                replacement = left[2] if left_len >= right_len else right[2]
        output[start:end] = replacement
    return output


def postprocess_predictions(preds, probs, num_classes, method, window, min_run):
    if method == "none":
        processed = preds.copy()
    elif method == "majority":
        processed = apply_majority_vote(preds, window, num_classes)
    elif method == "prob_avg":
        smoothed_probs = apply_prob_average(probs, window)
        processed = smoothed_probs.argmax(axis=1)
    else:
        raise ValueError(f"Unknown postprocess method: {method}")

    if min_run > 1:
        processed = enforce_min_run(processed, min_run)

    return processed


def compute_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "kappa": kappa,
    }


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_class_weight_grids(value, num_classes):
    parts = [p.strip() for p in value.split("|") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(f"--class_weight_grids needs {num_classes} parts, got {len(parts)}")
    grids = [parse_float_list(p) for p in parts]
    if any(len(g) == 0 for g in grids):
        raise ValueError("--class_weight_grids contains empty grid")
    return grids


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def write_csv(path, rows):
    header = [
        "weights", "method", "window", "min_run",
        "accuracy", "f1_macro", "f1_weighted", "kappa",
        "delta_accuracy", "delta_f1_macro", "delta_f1_weighted", "delta_kappa",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Grid search class weights + post-processing")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "cnn", "sequence", "deep_resnet", "deep_resnet_large"],
        help="Model type (default: auto-detect)",
    )
    parser.add_argument("--cache", default="./cache", help="Cache directory")
    parser.add_argument("--stages", type=int, default=3, choices=[3, 5], help="Number of stages")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio")
    parser.add_argument("--seq_len", type=int, default=0, help="Sequence length (0=auto detect, >0 for sequence model)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument(
        "--weight_grid",
        default="0.8,1.0,1.2,1.4,1.6",
        help="Comma-separated weight grid applied to all classes (if class_weight_grids not set)",
    )
    parser.add_argument(
        "--class_weight_grids",
        default="",
        help="Per-class grids separated by '|' (e.g. '1.0|0.9,1.0,1.1|1.0,1.2')",
    )
    parser.add_argument(
        "--methods",
        default="majority,prob_avg,none",
        help="Comma-separated methods: majority,prob_avg,none",
    )
    parser.add_argument(
        "--windows",
        default="3,5,7",
        help="Comma-separated odd window sizes (ignored for method=none)",
    )
    parser.add_argument(
        "--min_runs",
        default="1,2,3",
        help="Comma-separated min_run values (>=1)",
    )
    parser.add_argument(
        "--sort",
        default="f1_macro",
        choices=["f1_macro", "f1_weighted", "accuracy", "kappa"],
        help="Metric to sort by",
    )
    parser.add_argument("--top_k", type=int, default=20, help="Show top K results")
    parser.add_argument("--out_csv", default="", help="Optional CSV output path")
    args = parser.parse_args()

    methods = parse_str_list(args.methods)
    windows = parse_int_list(args.windows)
    min_runs = parse_int_list(args.min_runs)

    if any(m not in {"majority", "prob_avg", "none"} for m in methods):
        raise ValueError(f"Invalid methods: {methods}")
    if any(m < 1 for m in min_runs):
        raise ValueError("--min_runs must be >= 1")
    windows_odd = [w for w in windows if w > 0 and w % 2 == 1]
    if not windows_odd and any(m in {"majority", "prob_avg"} for m in methods):
        raise ValueError("No valid odd --windows provided")
    if len(windows_odd) != len(windows):
        dropped = sorted(set(windows) - set(windows_odd))
        print(f"  Dropping invalid windows (must be odd): {dropped}")

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

    if any(k.startswith("module.") for k in state_dict.keys()):
        print("  Removing DataParallel prefix from state_dict...")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("  Removing torch.compile() prefix from state_dict...")
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    if args.model_type == "auto":
        has_stage4 = any("stage4" in k for k in state_dict.keys())
        has_stage3_only = any("stage3" in k for k in state_dict.keys()) and not has_stage4
        is_sequence_model = any("backbone" in k or "temporal_lstm" in k for k in state_dict.keys())
        is_deep_resnet_large = has_stage4 and any("stem" in k for k in state_dict.keys()) and not is_sequence_model
        is_deep_resnet = has_stage3_only and any("stem" in k for k in state_dict.keys()) and not is_sequence_model

        if is_deep_resnet_large:
            model_type = "deep_resnet_large"
        elif is_deep_resnet:
            model_type = "deep_resnet"
        elif is_sequence_model:
            model_type = "sequence"
        else:
            model_type = "cnn"
        print(f"  Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type
        print(f"  Using specified model type: {model_type}")

    if model_type == "deep_resnet_large":
        seq_len = 0
        model = DeepSleepResNetLarge(num_classes=num_classes)
    elif model_type == "deep_resnet":
        seq_len = 0
        model = DeepSleepResNet(num_classes=num_classes)
    elif model_type == "sequence":
        seq_len = args.seq_len if args.seq_len > 0 else 11
        model = SleepSequenceModel(num_classes=num_classes, hidden_dim=256, seq_len=seq_len)
    else:
        seq_len = 0
        model = get_model(model_type="cnn", num_classes=num_classes)

    is_sequence_model = model_type == "sequence"
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
        print("\n  Converting to 3-stage (Wake/NREM/REM)...")
        labels = convert_to_3stage(labels)

    labels, label_map = remap_labels(labels)
    print(f"  Label mapping: {label_map}")
    print(f"  Classes: {len(label_map)}")
    print(f"  Stage names: {stage_names}")

    print(f"\n[3/4] Splitting data (seed=42, test_ratio={args.test_ratio})...")
    np.random.seed(42)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * args.test_ratio)

    test_indices = indices[:test_size]

    test_features = features[test_indices]
    test_labels = labels[test_indices]

    print(f"  Test samples: {len(test_labels)}")

    print("\n  Standardizing features (robust)...")
    test_features_norm = standardize_features(test_features, robust=True)

    print("\n[4/4] Evaluating (single pass)...")
    if is_sequence_model:
        _, probabilities, test_labels = evaluate_sequence(
            model,
            test_features_norm,
            test_labels,
            seq_len=seq_len,
            device=device,
            batch_size=args.batch_size,
        )
    else:
        _, probabilities = evaluate_batch(
            model,
            test_features_norm,
            device=device,
            batch_size=args.batch_size,
        )

    raw_predictions = probabilities.argmax(axis=1)
    raw_metrics = compute_metrics(test_labels, raw_predictions)
    print("\nRaw metrics")
    print(f"  Accuracy:      {raw_metrics['accuracy']*100:.1f}%")
    print(f"  F1 (macro):    {raw_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {raw_metrics['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa: {raw_metrics['kappa']:.4f}")

    if args.class_weight_grids:
        weight_grids = parse_class_weight_grids(args.class_weight_grids, num_classes)
    else:
        grid = parse_float_list(args.weight_grid)
        if len(grid) == 0:
            raise ValueError("--weight_grid is empty")
        weight_grids = [grid for _ in range(num_classes)]

    if any(any(w <= 0 for w in g) for g in weight_grids):
        raise ValueError("class weights must be > 0")

    weight_combos = list(itertools.product(*weight_grids))
    combos = []
    for weights in weight_combos:
        for method in methods:
            if method == "none":
                for min_run in min_runs:
                    combos.append((weights, method, 1, min_run))
            else:
                for window in windows_odd:
                    for min_run in min_runs:
                        combos.append((weights, method, window, min_run))

    print(f"\nSearching {len(combos)} combinations...")

    results = []
    for weights, method, window, min_run in combos:
        weighted_probs = apply_class_weights(probabilities, weights)
        weighted_preds = weighted_probs.argmax(axis=1)
        post_preds = postprocess_predictions(
            weighted_preds,
            weighted_probs,
            num_classes=num_classes,
            method=method,
            window=window,
            min_run=min_run,
        )
        metrics = compute_metrics(test_labels, post_preds)
        row = {
            "weights": "|".join(f"{w:.4f}" for w in weights),
            "method": method,
            "window": window,
            "min_run": min_run,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "kappa": metrics["kappa"],
            "delta_accuracy": metrics["accuracy"] - raw_metrics["accuracy"],
            "delta_f1_macro": metrics["f1_macro"] - raw_metrics["f1_macro"],
            "delta_f1_weighted": metrics["f1_weighted"] - raw_metrics["f1_weighted"],
            "delta_kappa": metrics["kappa"] - raw_metrics["kappa"],
        }
        results.append(row)

    results.sort(key=lambda r: r[args.sort], reverse=True)

    print("\nTop results")
    header = (
        "weights                         method    window  min_run  "
        "acc    f1_macro  f1_weighted  kappa   "
        "d_acc  d_f1_macro  d_f1_weighted  d_kappa"
    )
    print(header)
    print("-" * len(header))
    for row in results[:max(args.top_k, 1)]:
        print(
            f"{row['weights']:<30} "
            f"{row['method']:<8} {row['window']:>6} {row['min_run']:>7}  "
            f"{row['accuracy']*100:>5.1f}  {row['f1_macro']:>8.4f}  {row['f1_weighted']:>11.4f}  {row['kappa']:>6.4f}  "
            f"{row['delta_accuracy']*100:>+5.1f}  {row['delta_f1_macro']:>+10.4f}  {row['delta_f1_weighted']:>+13.4f}  {row['delta_kappa']:>+8.4f}"
        )

    if args.out_csv:
        write_csv(args.out_csv, results)
        print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
