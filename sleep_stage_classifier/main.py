import argparse
from pathlib import Path
import numpy as np

from .config import AudioConfig, ModelConfig, TrainConfig, DataConfig, SLEEP_STAGE_NAMES
from .data import PSGDataProcessor, create_data_loaders
from .models import get_model
from .train import Trainer, compute_class_weights


def main():
    parser = argparse.ArgumentParser(description="Sleep Stage Classification from Audio")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory with EDF/RML files")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "crnn", "transformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_librosa", action="store_true", help="Use librosa for feature extraction")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    args = parser.parse_args()
    
    audio_config = AudioConfig()
    model_config = ModelConfig()
    train_config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    data_config = DataConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    print("=" * 60)
    print("Sleep Stage Classification Pipeline")
    print("=" * 60)
    
    print("\n[1/4] Processing data...")
    processor = PSGDataProcessor(
        audio_config=audio_config,
        use_librosa=args.use_librosa,
        cache_dir=None if args.no_cache else data_config.cache_dir
    )
    
    features, labels = processor.process_subject(args.data_dir)
    
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        stage_name = SLEEP_STAGE_NAMES.get(u, f"Unknown_{u}")
        print(f"    {stage_name}: {c} ({100*c/len(labels):.1f}%)")
    
    print("\n[2/4] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        features, 
        labels,
        train_ratio=train_config.train_ratio,
        val_ratio=train_config.val_ratio,
        batch_size=train_config.batch_size,
        num_workers=0
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    print(f"\n[3/4] Building {args.model_type.upper()} model...")
    num_classes = len(unique)
    model = get_model(
        model_type=args.model_type,
        input_channels=1,
        num_classes=num_classes,
        hidden_dim=model_config.hidden_dim,
        dropout=model_config.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    class_weights = compute_class_weights(labels)
    print(f"  Class weights: {class_weights[:num_classes]}")
    
    print(f"\n[4/4] Training...")
    trainer = Trainer(
        model=model,
        train_config=train_config,
        class_weights=class_weights[:num_classes]
    )
    
    history = trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    metrics = trainer.get_detailed_metrics(test_loader)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
    
    print("\nClassification Report:")
    report = metrics['classification_report']
    for key, values in report.items():
        if isinstance(values, dict):
            print(f"  {key}:")
            print(f"    Precision: {values['precision']:.3f}")
            print(f"    Recall: {values['recall']:.3f}")
            print(f"    F1-score: {values['f1-score']:.3f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    output_path = Path(data_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / f"sleep_stage_{args.model_type}.pt"
    trainer.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


def run_demo():
    import torch
    
    audio_config = AudioConfig()
    model_config = ModelConfig()
    
    print("Sleep Stage Classification Demo")
    print("-" * 40)
    
    print("\n1. Processing sample data...")
    processor = PSGDataProcessor(audio_config=audio_config, use_librosa=False)
    
    try:
        features, labels = processor.process_subject(".")
        print(f"   Features: {features.shape}")
        print(f"   Labels: {labels.shape}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Generating synthetic data for demo...")
        features = np.random.randn(100, 128, 938).astype(np.float32)
        labels = np.random.randint(0, 4, 100)
    
    print("\n2. Creating model...")
    num_classes = len(np.unique(labels))
    model = get_model(
        model_type="cnn",
        num_classes=num_classes,
        hidden_dim=model_config.hidden_dim
    )
    print(f"   Model created with {num_classes} classes")
    
    print("\n3. Testing forward pass...")
    sample_input = torch.FloatTensor(features[:4]).unsqueeze(1)
    with torch.no_grad():
        output = model(sample_input)
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n4. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        features, labels, batch_size=16, num_workers=0
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
