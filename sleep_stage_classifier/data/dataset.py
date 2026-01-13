from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pickle

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

from .edf_reader import EDFReader, find_edf_files
from .rml_parser import RMLParser, find_rml_file
from ..features.audio_features import get_feature_extractor
from ..config import SLEEP_STAGE_MAP, AudioConfig


class SleepStageDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SleepStageDataset")
        
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
        if self.features.dim() == 3:
            self.features = self.features.unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class PSGDataProcessor:
    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        mic_channel_names: Optional[List[str]] = None,
        use_librosa: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.audio_config = audio_config or AudioConfig()
        self.edf_reader = EDFReader(mic_channel_names)
        self.use_librosa = use_librosa
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._feature_extractor = None
        
    def _get_feature_extractor(self, sample_rate: int):
        if self._feature_extractor is None or self._feature_extractor.sample_rate != sample_rate:
            self._feature_extractor = get_feature_extractor(
                use_librosa=self.use_librosa,
                sample_rate=sample_rate,
                n_mels=self.audio_config.n_mels,
                n_fft=min(self.audio_config.n_fft, sample_rate),
                hop_length=min(self.audio_config.hop_length, sample_rate // 4),
                n_mfcc=self.audio_config.n_mfcc
            )
        return self._feature_extractor
        
    def process_subject(
        self, 
        edf_dir: str,
        rml_dir: str,
        subject_id: str,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{subject_id}_features.pkl"
            if cache_path.exists():
                if verbose:
                    print(f"    Loading {subject_id} from cache...")
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                return cached['features'], cached['labels']
        
        edf_files = find_edf_files(edf_dir, subject_id)
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for {subject_id}")
        
        rml_file = find_rml_file(rml_dir, subject_id)
        
        if verbose:
            print(f"    Loading {len(edf_files)} EDF files for {subject_id}...")
        
        audio_segments = []
        orig_sr = None
        for i, edf_file in enumerate(sorted(edf_files)):
            if verbose:
                print(f"      [{i+1}/{len(edf_files)}] {Path(edf_file).name}")
            seg_audio, seg_sr = self.edf_reader.read_mic_channel(edf_file)
            if orig_sr is None:
                orig_sr = seg_sr
            audio_segments.append(seg_audio)
        
        audio = np.concatenate(audio_segments)
        if verbose:
            print(f"    Total audio: {len(audio)/orig_sr:.1f}s at {orig_sr}Hz")
        
        if verbose:
            print("    Parsing sleep stages...")
        rml_parser = RMLParser()
        rml_parser.parse(rml_file)
        
        total_duration = len(audio) / orig_sr
        stage_labels = rml_parser.get_labels_at_intervals(
            epoch_duration=self.audio_config.epoch_duration,
            total_duration=total_duration
        )
        
        if verbose:
            print(f"    Extracting features from {len(stage_labels)} epochs...")
        
        feature_extractor = self._get_feature_extractor(orig_sr)
        mel_features, _ = feature_extractor.extract_epoch_features(
            audio, 
            epoch_duration=self.audio_config.epoch_duration
        )
        
        num_epochs = min(len(mel_features), len(stage_labels))
        mel_features = mel_features[:num_epochs]
        stage_labels = stage_labels[:num_epochs]
        
        numeric_labels = np.array([
            SLEEP_STAGE_MAP.get(label, -1) for label in stage_labels
        ])
        
        valid_mask = numeric_labels >= 0
        mel_features = mel_features[valid_mask]
        numeric_labels = numeric_labels[valid_mask]
        
        if cache_path and self.cache_dir:
            if verbose:
                print("    Saving to cache...")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'features': mel_features,
                    'labels': numeric_labels
                }, f)
        
        return mel_features, numeric_labels
    
    def process_multiple_subjects(
        self,
        edf_dir: str,
        rml_dir: str,
        subject_ids: List[str],
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []
        
        for i, subject_id in enumerate(subject_ids):
            if verbose:
                print(f"\n[{i+1}/{len(subject_ids)}] Processing {subject_id}...")
            try:
                features, labels = self.process_subject(edf_dir, rml_dir, subject_id, verbose)
                all_features.append(features)
                all_labels.append(labels)
                if verbose:
                    print(f"    => {len(labels)} epochs extracted")
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if not all_features:
            raise ValueError("No data could be processed")
        
        return np.concatenate(all_features), np.concatenate(all_labels)


def create_data_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    random_seed: int = 42
):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for create_data_loaders")
    
    np.random.seed(random_seed)
    
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    split_info = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "random_seed": random_seed
    }
    
    train_dataset = SleepStageDataset(features[train_indices], labels[train_indices])
    val_dataset = SleepStageDataset(features[val_indices], labels[val_indices])
    test_dataset = SleepStageDataset(features[test_indices], labels[test_indices])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, split_info
