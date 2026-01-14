from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    epoch_duration: int = 30
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 40
    

@dataclass
class ModelConfig:
    input_channels: int = 1
    num_classes: int = 4
    hidden_dim: int = 128
    dropout: float = 0.3
    

@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4
    device: str = "cuda"
    

@dataclass
class DataConfig:
    data_dir: str = "."
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    mic_channel_names: List[str] = field(default_factory=lambda: ["Mic", "MSnore", "Snore", "Audio", "Sound"])
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


SLEEP_STAGE_MAP = {
    "Wake": 0,
    "NonREM1": 1,
    "NonREM2": 2,
    "NonREM3": 3,
    "REM": 4,
    "NotScored": -1,
}

SLEEP_STAGE_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2", 
    3: "N3",
    4: "REM",
}
