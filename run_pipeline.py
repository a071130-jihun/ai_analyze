#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from sleep_stage_classifier.config import AudioConfig, ModelConfig, TrainConfig, SLEEP_STAGE_NAMES
from sleep_stage_classifier.data.edf_reader import find_subject_ids
from sleep_stage_classifier.data.dataset import PSGDataProcessor, SleepStageDataset
from sleep_stage_classifier.models.classifier import get_model
from sleep_stage_classifier.train import Trainer, compute_class_weights

EDF_DIR = "./APNEA_EDF"
RML_DIR = "./APNEA_RML"
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def find_subject_ids_from_cache(cache_dir: str):
    """캐시 폴더에서 subject ID 추출"""
    import glob
    cache_files = glob.glob(os.path.join(cache_dir, "*_features.pkl"))
    subject_ids = []
    for f in cache_files:
        basename = os.path.basename(f)
        subject_id = basename.replace("_features.pkl", "")
        subject_ids.append(subject_id)
    return sorted(subject_ids)


def remap_labels_continuous(labels: np.ndarray):
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped = np.array([label_map[l] for l in labels])
    return remapped, label_map


def convert_to_3stage(labels: np.ndarray):
    """5단계 → 3단계 변환 (논문 기반)
    Wake(0) → 0 (Wake)
    N1(1), N2(2), N3(3) → 1 (NREM)
    REM(4) → 2 (REM)
    """
    mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    converted = np.array([mapping.get(l, 1) for l in labels])
    return converted


STAGE_NAMES_3 = {0: "Wake", 1: "NREM", 2: "REM"}


def split_data(features, labels, test_ratio=0.2, random_seed=42):
    np.random.seed(random_seed)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    
    test_size = int(n_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return {
        "train_features": features[train_indices],
        "train_labels": labels[train_indices],
        "test_features": features[test_indices],
        "test_labels": labels[test_indices],
        "train_indices": train_indices,
        "test_indices": test_indices
    }


def standardize_features(train_features, val_features=None, test_features=None):
    mean = train_features.mean(axis=(0, 1, 2), keepdims=True)
    std = train_features.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    
    train_norm = (train_features - mean) / std
    
    result = {"train": train_norm, "mean": mean, "std": std}
    
    if val_features is not None:
        result["val"] = (val_features - mean) / std
    if test_features is not None:
        result["test"] = (test_features - mean) / std
    
    return result


def create_loaders(
    features, 
    labels, 
    batch_size=16, 
    val_ratio=0.15, 
    use_augmentation=True, 
    use_balanced_sampling=True,
    aug_strength="medium",
    oversample_factor=1.0,
    random_seed=42
):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from sleep_stage_classifier.augmentation import get_train_transform
    
    np.random.seed(random_seed)
    
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * val_ratio)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    train_features = features[train_idx]
    val_features = features[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    
    normalized = standardize_features(train_features, val_features)
    train_features_norm = normalized["train"]
    val_features_norm = normalized["val"]
    
    print(f"  Feature stats - Train mean: {normalized['mean'].mean():.4f}, std: {normalized['std'].mean():.4f}")
    
    train_transform = get_train_transform(strength=aug_strength) if use_augmentation else None
    train_dataset = SleepStageDataset(train_features_norm, train_labels, transform=train_transform)
    val_dataset = SleepStageDataset(val_features_norm, val_labels)
    
    if use_balanced_sampling:
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.min()
        
        if oversample_factor > 1.0:
            class_weights = class_weights ** oversample_factor
        
        sample_weights = class_weights[train_labels]
        
        num_samples = len(train_labels)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        print(f"  Augmentation: {aug_strength}, Balanced sampling: ON")
        print(f"  Class weights: {dict(enumerate(class_weights[:len(class_counts)].round(2)))}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  Train samples: {len(train_labels)}, Val samples: {len(val_labels)}")
    
    return train_loader, val_loader, normalized["mean"], normalized["std"]


def evaluate_model(model, test_features, test_labels, device="cpu", batch_size=256):
    import torch
    
    model.eval()
    n_samples = len(test_features)
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = test_features[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
            outputs = model(batch_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_predictions.extend(preds)
            del batch_tensor, outputs
            if device != "cpu":
                torch.cuda.empty_cache()
    
    predictions = np.array(all_predictions)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(test_labels, predictions)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "kappa": kappa,
        "predictions": predictions,
        "labels": test_labels
    }


def plot_results(metrics, history, save_path="./results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    predictions = metrics["predictions"]
    labels = metrics["labels"]
    
    present_classes = np.unique(np.concatenate([labels, predictions]))
    target_names = [STAGE_NAMES.get(i, f"Class_{i}") for i in present_classes]
    
    cm = confusion_matrix(labels, predictions, labels=present_classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=axes[0, 0], cmap='Blues', colorbar=False)
    axes[0, 0].set_title('Confusion Matrix (Counts)')
    
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=target_names)
    disp2.plot(ax=axes[0, 1], cmap='Blues', colorbar=False, values_format='.2f')
    axes[0, 1].set_title('Confusion Matrix (Normalized)')
    
    axes[1, 0].bar(["Accuracy", "F1 (macro)", "F1 (weighted)", "Kappa"],
                   [metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"], metrics["kappa"]],
                   color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c'])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Evaluation Metrics (Test Set)')
    for i, v in enumerate([metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"], metrics["kappa"]]):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    if history and 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[1, 1].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in history:
            axes[1, 1].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].set_title('Training History')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nResults saved to: {save_path}")


def run_pipeline(
    edf_dir: str = EDF_DIR,
    rml_dir: str = RML_DIR,
    model_type: str = "cnn",
    epochs: int = 30,
    test_ratio: float = 0.2,
    batch_size: int = 16,
    num_stages: int = 5,
    aug_strength: str = "medium",
    oversample_factor: float = 1.0
):
    import torch
    
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    stage_names = STAGE_NAMES if num_stages == 5 else STAGE_NAMES_3
    
    print("=" * 70)
    print("  SLEEP STAGE CLASSIFICATION PIPELINE")
    print("  Train:Test = 8:2 Split with Automatic Validation")
    print("=" * 70)
    
    print("\n[1/6] Loading data...")
    cache_dir = "./cache"
    processor = PSGDataProcessor(
        audio_config=AudioConfig(),
        use_librosa=False,
        cache_dir=cache_dir
    )
    
    if os.path.exists(edf_dir):
        subject_ids = find_subject_ids(edf_dir)
        print(f"  Found {len(subject_ids)} subject(s) from EDF directory")
    else:
        subject_ids = find_subject_ids_from_cache(cache_dir)
        print(f"  Found {len(subject_ids)} subject(s) from cache (EDF dir not found)")
    
    if len(subject_ids) == 0:
        print("  ERROR: No subjects found. Need either EDF files or cache.")
        return
    
    if len(subject_ids) == 1:
        features, labels = processor.process_subject(
            edf_dir, rml_dir, subject_ids[0], verbose=True
        )
    else:
        features, labels = processor.process_multiple_subjects(
            edf_dir, rml_dir, subject_ids, verbose=True
        )
    
    print(f"\n  Total samples: {len(labels)}")
    print(f"  Feature shape: {features[0].shape}")
    
    print("\n[2/6] Preprocessing labels...")
    unique_orig = np.unique(labels)
    print(f"  Original labels (raw): {list(unique_orig)}")
    print(f"  Original distribution:")
    for lbl in unique_orig:
        count = np.sum(labels == lbl)
        print(f"    Label {lbl}: {count} ({100*count/len(labels):.1f}%)")
    
    if num_stages == 3:
        print(f"\n  Converting to 3-stage: Wake(0)/NREM(1)/REM(2)")
        print(f"    Mapping: 0->0(Wake), 1,2,3->1(NREM), 4->2(REM)")
        labels = convert_to_3stage(labels)
        
        print(f"  After 3-stage conversion:")
        for lbl in [0, 1, 2]:
            count = np.sum(labels == lbl)
            name = STAGE_NAMES_3.get(lbl, f"Class_{lbl}")
            print(f"    {name}({lbl}): {count} ({100*count/len(labels):.1f}%)")
    
    labels, label_map = remap_labels_continuous(labels)
    num_classes = len(np.unique(labels))
    
    print(f"\n  Final label mapping: {label_map}")
    print(f"  Number of classes: {num_classes}")
    
    print(f"\n  Final class distribution:")
    for old_label, new_label in label_map.items():
        count = np.sum(labels == new_label)
        name = stage_names.get(old_label, f"Class_{old_label}")
        print(f"    {name} (label={new_label}): {count} samples ({100*count/len(labels):.1f}%)")
    
    print(f"\n[3/6] Splitting data (Train {int((1-test_ratio)*100)}% : Test {int(test_ratio*100)}%)...")
    data = split_data(features, labels, test_ratio=test_ratio)
    
    print(f"  Train set: {len(data['train_labels'])} samples")
    print(f"  Test set:  {len(data['test_labels'])} samples")
    
    print(f"\n[4/6] Building {model_type.upper()} model...")
    model = get_model(
        model_type=model_type,
        input_channels=1,
        num_classes=num_classes,
        hidden_dim=128,
        dropout=0.3
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    
    print(f"\n[5/6] Training for {epochs} epochs...")
    train_loader, val_loader, feat_mean, feat_std = create_loaders(
        data["train_features"], 
        data["train_labels"],
        batch_size=batch_size,
        val_ratio=0.15,
        aug_strength=aug_strength,
        oversample_factor=oversample_factor
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    class_weights = compute_class_weights(data["train_labels"])[:num_classes]
    
    train_config = TrainConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        device=device
    )
    
    trainer = Trainer(
        model=model,
        train_config=train_config,
        class_weights=class_weights
    )
    
    print()
    history = trainer.train(train_loader, val_loader, num_epochs=epochs)
    
    os.makedirs("./output", exist_ok=True)
    model_path = f"./output/sleep_stage_{model_type}.pt"
    trainer.save_model(model_path)
    print(f"\n  Model saved: {model_path}")
    
    print(f"\n[6/6] Evaluating on TEST SET...")
    test_features_norm = (data["test_features"] - feat_mean) / feat_std
    metrics = evaluate_model(model, test_features_norm, data["test_labels"], device)
    
    print("\n" + "=" * 70)
    print("  FINAL RESULTS (TEST SET - NOT SEEN DURING TRAINING)")
    print("=" * 70)
    print(f"\n  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa:   {metrics['kappa']:.4f}")
    
    print("\n  Classification Report:")
    present_classes = np.unique(np.concatenate([data["test_labels"], metrics["predictions"]]))
    target_names = [stage_names.get(i, f"Class_{i}") for i in present_classes]
    print(classification_report(
        data["test_labels"], 
        metrics["predictions"],
        labels=present_classes,
        target_names=target_names,
        zero_division=0
    ))
    
    os.makedirs("./output", exist_ok=True)
    
    model_path = f"./output/sleep_stage_{model_type}.pt"
    trainer.save_model(model_path)
    print(f"  Model saved: {model_path}")
    
    np.savez(
        "./output/evaluation_data.npz",
        features=features,
        labels=labels,
        test_indices=data["test_indices"],
        train_indices=data["train_indices"],
        test_predictions=metrics["predictions"],
        test_labels=data["test_labels"]
    )
    print(f"  Data saved: ./output/evaluation_data.npz")
    
    plot_results(metrics, history, "./output/results.png")
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    
    return model, metrics, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sleep Stage Classification Pipeline")
    parser.add_argument("--edf_dir", default=EDF_DIR)
    parser.add_argument("--rml_dir", default=RML_DIR)
    parser.add_argument("--model", default="cnn", choices=["cnn", "crnn", "transformer"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio (default: 0.2 = 20%%)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stages", type=int, default=5, choices=[3, 5],
                        help="Number of sleep stages: 3 (Wake/NREM/REM) or 5 (Wake/N1/N2/N3/REM)")
    parser.add_argument("--aug", default="medium", choices=["none", "light", "medium", "strong", "aggressive"],
                        help="Augmentation strength (default: medium, use 'none' to disable)")
    parser.add_argument("--oversample", type=float, default=1.0,
                        help="Oversample factor for minority classes (default: 1.0, try 1.5-2.0)")
    args = parser.parse_args()
    
    run_pipeline(
        edf_dir=args.edf_dir,
        rml_dir=args.rml_dir,
        model_type=args.model,
        epochs=args.epochs,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        num_stages=args.stages,
        aug_strength=args.aug,
        oversample_factor=args.oversample
    )
