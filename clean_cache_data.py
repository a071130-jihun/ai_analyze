#!/usr/bin/env python3
"""
캐시 데이터 정제 스크립트
- 전환 구간(transition) 제거
- 짧은 연속 구간 제거 (노이즈)
- Feature 이상치 제거
"""
import os
import glob
import pickle
import numpy as np
from collections import Counter

CACHE_DIR = "./cache"
CLEAN_CACHE_DIR = "./cache_clean"
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def load_cache(cache_dir):
    """캐시 파일들 로드"""
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*_features.pkl")))
    all_data = []
    
    for cache_file in cache_files:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        subject_id = os.path.basename(cache_file).replace("_features.pkl", "")
        all_data.append({
            'subject_id': subject_id,
            'features': data['features'],
            'labels': data['labels'],
            'file_path': cache_file
        })
        print(f"  Loaded: {subject_id} ({len(data['labels'])} epochs)")
    
    return all_data


def analyze_labels(labels, subject_id=""):
    """라벨 분포 및 전환 분석"""
    print(f"\n  [{subject_id}] Label Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = STAGE_NAMES.get(u, f"Class_{u}")
        print(f"    {name}: {c} ({100*c/len(labels):.1f}%)")
    
    transitions = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            transitions += 1
    print(f"    Transitions: {transitions} ({100*transitions/len(labels):.1f}%)")
    
    return transitions


def find_transition_indices(labels, window=1):
    """전환 구간 인덱스 찾기 (앞뒤 window개 epoch 포함)"""
    transition_indices = set()
    
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            for j in range(max(0, i - window), min(len(labels), i + window + 1)):
                transition_indices.add(j)
    
    return sorted(transition_indices)


def find_short_segments(labels, min_length=3):
    """짧은 연속 구간 찾기 (min_length 미만)"""
    short_indices = set()
    
    i = 0
    while i < len(labels):
        current_label = labels[i]
        segment_start = i
        
        while i < len(labels) and labels[i] == current_label:
            i += 1
        
        segment_length = i - segment_start
        
        if segment_length < min_length:
            for j in range(segment_start, i):
                short_indices.add(j)
    
    return sorted(short_indices)


def find_feature_outliers(features, percentile=1):
    """Feature 이상치 찾기"""
    feature_means = features.mean(axis=(1, 2))
    
    low = np.percentile(feature_means, percentile)
    high = np.percentile(feature_means, 100 - percentile)
    
    outlier_indices = np.where((feature_means < low) | (feature_means > high))[0]
    
    return list(outlier_indices)


def clean_subject_data(data, remove_transitions=True, transition_window=1,
                       remove_short=True, min_segment_length=3,
                       remove_outliers=True, outlier_percentile=1):
    """단일 피험자 데이터 정제"""
    features = data['features']
    labels = data['labels']
    subject_id = data['subject_id']
    
    n_original = len(labels)
    remove_indices = set()
    
    if remove_transitions:
        trans_idx = find_transition_indices(labels, window=transition_window)
        remove_indices.update(trans_idx)
        print(f"    Transition epochs: {len(trans_idx)}")
    
    if remove_short:
        short_idx = find_short_segments(labels, min_length=min_segment_length)
        remove_indices.update(short_idx)
        print(f"    Short segment epochs: {len(short_idx)}")
    
    if remove_outliers:
        outlier_idx = find_feature_outliers(features, percentile=outlier_percentile)
        remove_indices.update(outlier_idx)
        print(f"    Feature outliers: {len(outlier_idx)}")
    
    keep_indices = [i for i in range(n_original) if i not in remove_indices]
    
    clean_features = features[keep_indices]
    clean_labels = labels[keep_indices]
    
    n_removed = n_original - len(keep_indices)
    print(f"    Removed: {n_removed}/{n_original} ({100*n_removed/n_original:.1f}%)")
    print(f"    Kept: {len(keep_indices)} epochs")
    
    return {
        'subject_id': subject_id,
        'features': clean_features,
        'labels': clean_labels,
        'original_indices': np.array(keep_indices)
    }


def save_clean_cache(clean_data, output_dir):
    """정제된 캐시 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    for data in clean_data:
        output_file = os.path.join(output_dir, f"{data['subject_id']}_features.pkl")
        
        save_data = {
            'features': data['features'],
            'labels': data['labels'],
            'original_indices': data.get('original_indices', None)
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"  Saved: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean cache data")
    parser.add_argument("--cache", default=CACHE_DIR, help="Input cache directory")
    parser.add_argument("--output", default=CLEAN_CACHE_DIR, help="Output cache directory")
    parser.add_argument("--transition_window", type=int, default=1, help="Epochs to remove around transitions")
    parser.add_argument("--min_segment", type=int, default=3, help="Minimum segment length to keep")
    parser.add_argument("--outlier_pct", type=float, default=1.0, help="Outlier percentile to remove")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze, don't clean")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  CACHE DATA CLEANING")
    print("=" * 70)
    
    print(f"\n[1/3] Loading cache from: {args.cache}")
    all_data = load_cache(args.cache)
    
    print(f"\n[2/3] Analyzing data...")
    total_original = 0
    total_transitions = 0
    
    for data in all_data:
        total_original += len(data['labels'])
        total_transitions += analyze_labels(data['labels'], data['subject_id'])
    
    print(f"\n  Total epochs: {total_original}")
    print(f"  Total transitions: {total_transitions}")
    
    if args.analyze_only:
        print("\n  [Analyze only mode - not cleaning]")
        return
    
    print(f"\n[3/3] Cleaning data...")
    print(f"  Settings:")
    print(f"    - Remove transitions: window={args.transition_window}")
    print(f"    - Remove short segments: min_length={args.min_segment}")
    print(f"    - Remove outliers: percentile={args.outlier_pct}%")
    
    clean_data = []
    for data in all_data:
        print(f"\n  Processing: {data['subject_id']}")
        clean = clean_subject_data(
            data,
            transition_window=args.transition_window,
            min_segment_length=args.min_segment,
            outlier_percentile=args.outlier_pct
        )
        clean_data.append(clean)
    
    total_clean = sum(len(d['labels']) for d in clean_data)
    print(f"\n  Total after cleaning: {total_clean}/{total_original} ({100*total_clean/total_original:.1f}%)")
    
    print(f"\n  Saving to: {args.output}")
    save_clean_cache(clean_data, args.output)
    
    print("\n  Clean label distribution:")
    all_clean_labels = np.concatenate([d['labels'] for d in clean_data])
    unique, counts = np.unique(all_clean_labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = STAGE_NAMES.get(u, f"Class_{u}")
        print(f"    {name}: {c} ({100*c/len(all_clean_labels):.1f}%)")
    
    print("\n" + "=" * 70)
    print("  CLEANING COMPLETE")
    print(f"  Use --cache {args.output} for training")
    print("=" * 70)


if __name__ == "__main__":
    main()
