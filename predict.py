#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from sleep_stage_classifier.data.edf_reader import EDFReader, find_edf_files
from sleep_stage_classifier.features.audio_features import get_feature_extractor
from sleep_stage_classifier.models.classifier import get_model

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "REM"}


def load_trained_model(model_path: str, num_classes: int = 4):
    model = get_model(model_type="cnn", num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_from_edf(model, edf_path: str, epoch_duration: int = 30):
    reader = EDFReader()
    audio, sr = reader.read_mic_channel(edf_path)
    
    extractor = get_feature_extractor(
        use_librosa=False,
        sample_rate=sr,
        n_fft=min(512, sr),
        hop_length=min(128, sr // 4)
    )
    
    mel_features, _ = extractor.extract_epoch_features(audio, epoch_duration)
    
    features = torch.FloatTensor(mel_features).unsqueeze(1)
    
    with torch.no_grad():
        outputs = model(features)
        predictions = outputs.argmax(dim=1).numpy()
        probabilities = torch.softmax(outputs, dim=1).numpy()
    
    return predictions, probabilities


def main():
    model_path = "./output/sleep_stage_cnn.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run training first: python run_pipeline.py --mode train")
        return
    
    print("Loading model...")
    model = load_trained_model(model_path)
    print("Model loaded!\n")
    
    edf_files = find_edf_files("./APNEA_EDF", "00000999-100507")
    if not edf_files:
        print("No EDF files found")
        return
    
    print(f"Predicting from: {os.path.basename(edf_files[0])}")
    predictions, probabilities = predict_from_edf(model, edf_files[0])
    
    print(f"\nTotal epochs: {len(predictions)}")
    print(f"\n{'Epoch':<8} {'Prediction':<12} {'Confidence':<12} {'Probabilities'}")
    print("-" * 70)
    
    for i in range(min(20, len(predictions))):
        pred = predictions[i]
        prob = probabilities[i]
        conf = prob[pred] * 100
        prob_str = " ".join([f"{STAGE_NAMES[j]}:{p:.1%}" for j, p in enumerate(prob)])
        print(f"{i:<8} {STAGE_NAMES[pred]:<12} {conf:.1f}%        {prob_str}")
    
    if len(predictions) > 20:
        print(f"... ({len(predictions) - 20} more epochs)")
    
    print("\n--- Summary ---")
    unique, counts = np.unique(predictions, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / len(predictions)
        print(f"  {STAGE_NAMES[u]}: {c} epochs ({pct:.1f}%)")


if __name__ == "__main__":
    main()
