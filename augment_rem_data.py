#!/usr/bin/env python3
"""
REM Sleep Data Augmentation Script

Usage:
    python augment_rem_data.py --cache_dir ./cache --output_dir ./cache_augmented
    python augment_rem_data.py --cache_dir ./cache --target_ratio 0.15 --undersample_nrem
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from collections import Counter
from tqdm import tqdm


LABEL_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


class REMAugmentor:
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def time_shift(self, x: np.ndarray, max_shift: int = 10) -> np.ndarray:
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, shift, axis=-1)
    
    def freq_shift(self, x: np.ndarray, max_shift: int = 5) -> np.ndarray:
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, shift, axis=-2)
    
    def add_noise(self, x: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        noise = np.random.randn(*x.shape) * noise_level * np.std(x)
        return x + noise
    
    def time_stretch(self, x: np.ndarray, rate_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        rate = np.random.uniform(*rate_range)
        n_freq, n_time = x.shape[-2], x.shape[-1]
        new_time = int(n_time * rate)
        
        old_indices = np.linspace(0, n_time - 1, new_time)
        new_x = np.zeros_like(x)
        
        for i in range(n_freq):
            new_x[..., i, :] = np.interp(
                np.arange(n_time),
                np.linspace(0, n_time - 1, new_time),
                x[..., i, :int(new_time)] if new_time <= n_time else 
                np.interp(np.linspace(0, new_time - 1, n_time), np.arange(new_time), 
                          np.pad(x[..., i, :], (0, new_time - n_time), mode='edge'))
            )
        return new_x
    
    def amplitude_scale(self, x: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        scale = np.random.uniform(*scale_range)
        return x * scale
    
    def spec_augment(self, x: np.ndarray, freq_mask: int = 10, time_mask: int = 15) -> np.ndarray:
        x = x.copy()
        n_freq, n_time = x.shape[-2], x.shape[-1]
        
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, max(1, n_freq - f))
        x[..., f0:f0+f, :] = 0
        
        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, max(1, n_time - t))
        x[..., :, t0:t0+t] = 0
        
        return x
    
    def mixup(self, x1: np.ndarray, x2: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2
    
    def augment_single(self, x: np.ndarray, aug_type: str = "random") -> np.ndarray:
        if aug_type == "random":
            augmentations = [
                lambda x: self.time_shift(x, max_shift=8),
                lambda x: self.freq_shift(x, max_shift=3),
                lambda x: self.add_noise(x, noise_level=0.015),
                lambda x: self.amplitude_scale(x, (0.85, 1.15)),
                lambda x: self.spec_augment(x, freq_mask=8, time_mask=12),
            ]
            n_augs = np.random.randint(1, 4)
            selected = np.random.choice(len(augmentations), n_augs, replace=False)
            
            result = x.copy()
            for idx in selected:
                result = augmentations[idx](result)
            return result
        
        elif aug_type == "light":
            x = self.time_shift(x, max_shift=5)
            if np.random.random() > 0.5:
                x = self.add_noise(x, noise_level=0.01)
            return x
        
        elif aug_type == "strong":
            x = self.time_shift(x, max_shift=12)
            x = self.freq_shift(x, max_shift=5)
            x = self.add_noise(x, noise_level=0.025)
            x = self.spec_augment(x, freq_mask=15, time_mask=20)
            return x
        
        return x


def load_cache_files(cache_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    features_dict = {}
    labels_dict = {}
    
    cache_files = list(cache_dir.glob("*_features.pkl"))
    print(f"Found {len(cache_files)} cache files")
    
    for cache_file in tqdm(cache_files, desc="Loading cache"):
        subject_id = cache_file.stem.replace("_features", "")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        features_dict[subject_id] = data['features']
        labels_dict[subject_id] = data['labels']
    
    return features_dict, labels_dict


def analyze_distribution(labels_dict: Dict[str, np.ndarray]) -> Dict[int, int]:
    all_labels = np.concatenate(list(labels_dict.values()))
    counts = Counter(all_labels)
    total = len(all_labels)
    
    print("\n=== Current Distribution ===")
    for label in sorted(counts.keys()):
        name = LABEL_NAMES.get(label, f"Unknown({label})")
        pct = counts[label] / total * 100
        print(f"  {name}: {counts[label]:,} ({pct:.1f}%)")
    print(f"  Total: {total:,}")
    
    return dict(counts)


def augment_rem_data(
    features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    target_rem_ratio: float = 0.15,
    undersample_nrem: bool = False,
    nrem_target_ratio: float = 0.50,
    random_seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    np.random.seed(random_seed)
    augmentor = REMAugmentor(random_seed)
    
    all_features = []
    all_labels = []
    all_subjects = []
    
    for subject_id in features_dict:
        features = features_dict[subject_id]
        labels = labels_dict[subject_id]
        for i in range(len(labels)):
            all_features.append(features[i])
            all_labels.append(labels[i])
            all_subjects.append(subject_id)
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    counts = Counter(all_labels)
    total = len(all_labels)
    
    rem_indices = np.where(all_labels == 4)[0]
    rem_count = len(rem_indices)
    
    nrem_indices = np.where((all_labels >= 1) & (all_labels <= 3))[0]
    nrem_count = len(nrem_indices)
    
    wake_indices = np.where(all_labels == 0)[0]
    wake_count = len(wake_indices)
    
    print(f"\nCurrent: REM={rem_count} ({rem_count/total*100:.1f}%), "
          f"NREM={nrem_count} ({nrem_count/total*100:.1f}%), "
          f"Wake={wake_count} ({wake_count/total*100:.1f}%)")
    
    if undersample_nrem:
        target_nrem = int(total * nrem_target_ratio)
        expected_total = wake_count + target_nrem + rem_count
        target_rem_count = int(expected_total * target_rem_ratio / (1 - target_rem_ratio))
    else:
        target_rem_count = int((total - rem_count) * target_rem_ratio / (1 - target_rem_ratio))
    
    augment_count = max(0, target_rem_count - rem_count)
    
    print(f"\nTarget REM: {target_rem_count} (need to generate {augment_count} samples)")
    
    augmented_features = []
    augmented_labels = []
    
    if augment_count > 0 and rem_count > 0:
        rem_features = all_features[rem_indices]
        
        augs_per_sample = augment_count // rem_count + 1
        
        print(f"Augmenting each REM sample ~{augs_per_sample} times...")
        
        generated = 0
        aug_types = ["random", "light", "strong"]
        
        for i in tqdm(range(augment_count), desc="Generating REM samples"):
            src_idx = i % rem_count
            src_feature = rem_features[src_idx]
            
            aug_type = aug_types[i % len(aug_types)]
            
            if np.random.random() < 0.3 and rem_count > 1:
                other_idx = np.random.randint(0, rem_count)
                while other_idx == src_idx:
                    other_idx = np.random.randint(0, rem_count)
                src_feature = augmentor.mixup(src_feature, rem_features[other_idx])
            
            aug_feature = augmentor.augment_single(src_feature, aug_type)
            
            augmented_features.append(aug_feature)
            augmented_labels.append(4)
            generated += 1
        
        print(f"Generated {generated} augmented REM samples")
    
    final_features = list(all_features)
    final_labels = list(all_labels)
    
    if undersample_nrem and nrem_count > 0:
        new_total = total + len(augmented_features)
        target_nrem = int(new_total * nrem_target_ratio)
        
        if target_nrem < nrem_count:
            remove_count = nrem_count - target_nrem
            print(f"\nUndersampling NREM: removing {remove_count} samples")
            
            n1_indices = np.where(all_labels == 1)[0]
            n2_indices = np.where(all_labels == 2)[0]
            n3_indices = np.where(all_labels == 3)[0]
            
            for indices, name in [(n1_indices, "N1"), (n2_indices, "N2"), (n3_indices, "N3")]:
                if len(indices) > 0:
                    ratio = len(indices) / nrem_count
                    to_remove = int(remove_count * ratio)
                    if to_remove > 0:
                        remove_indices = np.random.choice(indices, min(to_remove, len(indices)), replace=False)
                        for idx in sorted(remove_indices, reverse=True):
                            final_features[idx] = None
                            final_labels[idx] = None
            
            final_features = [f for f in final_features if f is not None]
            final_labels = [l for l in final_labels if l is not None]
    
    final_features.extend(augmented_features)
    final_labels.extend(augmented_labels)
    
    final_features = np.array(final_features)
    final_labels = np.array(final_labels)
    
    shuffle_idx = np.random.permutation(len(final_labels))
    final_features = final_features[shuffle_idx]
    final_labels = final_labels[shuffle_idx]
    
    final_counts = Counter(final_labels)
    final_total = len(final_labels)
    
    print("\n=== Final Distribution ===")
    for label in sorted(final_counts.keys()):
        name = LABEL_NAMES.get(label, f"Unknown({label})")
        pct = final_counts[label] / final_total * 100
        print(f"  {name}: {final_counts[label]:,} ({pct:.1f}%)")
    print(f"  Total: {final_total:,}")
    
    return {"augmented": final_features}, {"augmented": final_labels}


def save_augmented_data(
    features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for subject_id in features_dict:
        output_path = output_dir / f"{subject_id}_features.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'features': features_dict[subject_id],
                'labels': labels_dict[subject_id]
            }, f)
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="REM Sleep Data Augmentation")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Input cache directory")
    parser.add_argument("--output_dir", type=str, default="./cache_augmented",
                        help="Output directory for augmented data")
    parser.add_argument("--target_ratio", type=float, default=0.15,
                        help="Target REM ratio (default: 0.15 = 15%%)")
    parser.add_argument("--undersample_nrem", action="store_true",
                        help="Also undersample NREM data")
    parser.add_argument("--nrem_ratio", type=float, default=0.50,
                        help="Target NREM ratio if undersampling (default: 0.50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze distribution, don't augment")
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        return
    
    print("Loading cache files...")
    features_dict, labels_dict = load_cache_files(cache_dir)
    
    if not features_dict:
        print("No cache files found!")
        return
    
    analyze_distribution(labels_dict)
    
    if args.analyze_only:
        return
    
    print("\n" + "="*50)
    print("Starting REM augmentation...")
    print("="*50)
    
    aug_features, aug_labels = augment_rem_data(
        features_dict,
        labels_dict,
        target_rem_ratio=args.target_ratio,
        undersample_nrem=args.undersample_nrem,
        nrem_target_ratio=args.nrem_ratio,
        random_seed=args.seed
    )
    
    print("\nSaving augmented data...")
    save_augmented_data(aug_features, aug_labels, output_dir)
    
    print("\nDone!")
    print(f"Augmented data saved to: {output_dir}")
    print(f"\nTo use augmented data, set cache_dir to: {output_dir}")


if __name__ == "__main__":
    main()
