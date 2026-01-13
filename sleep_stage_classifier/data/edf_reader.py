from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pyedflib


class EDFReader:
    def __init__(self, mic_channel_names: List[str] = None):
        self.mic_channel_names = mic_channel_names or ["Mic", "MSnore", "Snore", "Audio", "Sound"]
    
    def find_mic_channel(self, edf_path: str) -> Tuple[int, str]:
        with pyedflib.EdfReader(edf_path) as edf:
            signal_labels = edf.getSignalLabels()
            for idx, label in enumerate(signal_labels):
                label_clean = label.strip()
                for mic_name in self.mic_channel_names:
                    if mic_name.lower() in label_clean.lower():
                        return idx, label_clean
        raise ValueError(f"No mic channel found in {edf_path}. Available: {signal_labels}")
    
    def read_mic_channel(self, edf_path: str) -> Tuple[np.ndarray, int]:
        mic_idx, mic_label = self.find_mic_channel(edf_path)
        
        with pyedflib.EdfReader(edf_path) as edf:
            sample_rate = int(edf.getSampleFrequency(mic_idx))
            signal = edf.readSignal(mic_idx)
            
        return signal.astype(np.float32), sample_rate
    
    def read_all_edf_files(self, edf_paths: List[str]) -> Tuple[np.ndarray, int]:
        signals = []
        sample_rate = None
        
        sorted_paths = sorted(edf_paths, key=lambda x: x)
        
        for edf_path in sorted_paths:
            signal, sr = self.read_mic_channel(edf_path)
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise ValueError(f"Sample rate mismatch: {sample_rate} vs {sr}")
            signals.append(signal)
        
        if sample_rate is None:
            raise ValueError("No EDF files provided")
        
        return np.concatenate(signals), sample_rate
    
    def get_edf_info(self, edf_path: str) -> Dict:
        with pyedflib.EdfReader(edf_path) as edf:
            info = {
                "num_signals": edf.signals_in_file,
                "signal_labels": edf.getSignalLabels(),
                "sample_frequencies": [edf.getSampleFrequency(i) for i in range(edf.signals_in_file)],
                "duration": edf.getFileDuration(),
                "start_datetime": edf.getStartdatetime(),
            }
        return info


def find_edf_files(edf_dir: str, subject_id: str = None) -> List[str]:
    edf_path = Path(edf_dir)
    
    if subject_id:
        subject_dir = edf_path / subject_id
        if subject_dir.exists():
            edf_files = sorted(subject_dir.glob("*.edf"))
            return [str(f) for f in edf_files]
        
        edf_files = sorted(edf_path.glob(f"{subject_id}*.edf"))
        return [str(f) for f in edf_files]
    
    edf_files = sorted(edf_path.rglob("*.edf"))
    return [str(f) for f in edf_files]


def find_subject_ids(edf_dir: str) -> List[str]:
    edf_path = Path(edf_dir)
    subject_ids = []
    
    for subdir in edf_path.iterdir():
        if subdir.is_dir():
            edf_files = list(subdir.glob("*.edf"))
            if edf_files:
                subject_ids.append(subdir.name)
    
    return sorted(subject_ids)
