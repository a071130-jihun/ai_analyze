#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sleep_stage_classifier.data.edf_reader import EDFReader, find_edf_files, find_subject_ids
from sleep_stage_classifier.data.rml_parser import RMLParser, find_rml_file
from sleep_stage_classifier.features.audio_features import get_feature_extractor

EDF_DIR = "./APNEA_EDF"
RML_DIR = "./APNEA_RML"

STAGE_COLORS = {
    "Wake": "#e74c3c",
    "NonREM1": "#3498db", 
    "NonREM2": "#2ecc71",
    "NonREM3": "#9b59b6",
    "REM": "#f39c12",
    "NotScored": "#95a5a6",
}

STAGE_SHORT = {
    "Wake": "W",
    "NonREM1": "N1",
    "NonREM2": "N2", 
    "NonREM3": "N3",
    "REM": "REM",
    "NotScored": "?",
}


def visualize_subject(edf_dir: str, rml_dir: str, subject_id: str, save_path: str = None):
    print(f"Loading data for {subject_id}...")
    
    edf_files = find_edf_files(edf_dir, subject_id)
    rml_file = find_rml_file(rml_dir, subject_id)
    
    reader = EDFReader()
    
    print("  Reading audio from first EDF file...")
    audio, sr = reader.read_mic_channel(edf_files[0])
    
    print("  Parsing sleep stages...")
    parser = RMLParser()
    parser.parse(rml_file)
    stages = parser.stages
    total_duration = parser.total_duration
    
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(3, 1, 1)
    time_audio = np.arange(len(audio)) / sr
    
    downsample = max(1, len(audio) // 10000)
    ax1.plot(time_audio[::downsample], audio[::downsample], linewidth=0.5, color='#2c3e50')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Audio Waveform (Mic/Snore Channel) - First Hour\nSample Rate: {sr} Hz, Duration: {len(audio)/sr:.0f}s')
    ax1.set_xlim(0, len(audio)/sr)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 1, 2)
    
    for i, stage in enumerate(stages):
        start = stage.start_time
        if i + 1 < len(stages):
            end = stages[i + 1].start_time
        else:
            end = total_duration
        
        color = STAGE_COLORS.get(stage.stage_type, "#95a5a6")
        ax2.axvspan(start, end, alpha=0.7, color=color)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Sleep Stage')
    ax2.set_title(f'Sleep Stage Timeline - Total Duration: {total_duration:.0f}s ({total_duration/3600:.1f} hours)')
    ax2.set_xlim(0, total_duration)
    ax2.set_yticks([])
    
    legend_elements = [Patch(facecolor=color, label=f"{STAGE_SHORT[name]} ({name})") 
                       for name, color in STAGE_COLORS.items() if name != "NotScored"]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=5)
    
    ax3 = plt.subplot(3, 1, 3)
    
    extractor = get_feature_extractor(
        use_librosa=False,
        sample_rate=sr,
        n_fft=min(512, sr),
        hop_length=min(128, sr // 4),
        n_mels=64
    )
    
    epoch_duration = 30
    epoch_samples = epoch_duration * sr
    
    example_epochs = []
    example_stages = []
    for stage_type in ["Wake", "NonREM1", "NonREM2", "REM"]:
        for s in stages:
            if s.stage_type == stage_type:
                epoch_start = int(s.start_time)
                if epoch_start + epoch_duration <= len(audio) / sr:
                    example_epochs.append(epoch_start)
                    example_stages.append(stage_type)
                    break
    
    if example_epochs:
        n_examples = len(example_epochs)
        for idx, (epoch_start, stage_type) in enumerate(zip(example_epochs, example_stages)):
            ax_sub = plt.subplot(3, n_examples, 2 * n_examples + idx + 1)
            
            start_sample = epoch_start * sr
            end_sample = start_sample + epoch_samples
            epoch_audio = audio[start_sample:end_sample]
            
            epoch_audio = epoch_audio / (np.max(np.abs(epoch_audio)) + 1e-8)
            mel_spec = extractor.compute_mel_spectrogram(epoch_audio)
            
            ax_sub.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            ax_sub.set_title(f'{STAGE_SHORT[stage_type]}\n(t={epoch_start}s)')
            ax_sub.set_xlabel('Time frame')
            if idx == 0:
                ax_sub.set_ylabel('Mel band')
            else:
                ax_sub.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    else:
        save_path = "./data_visualization.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.close()
    
    print("\n--- Data Summary ---")
    print(f"Subject: {subject_id}")
    print(f"Audio sample rate: {sr} Hz")
    print(f"Total duration: {total_duration:.0f}s ({total_duration/3600:.2f} hours)")
    print(f"Number of stage annotations: {len(stages)}")
    
    stage_counts = {}
    for s in stages:
        stage_counts[s.stage_type] = stage_counts.get(s.stage_type, 0) + 1
    
    print("\nStage distribution:")
    for stage_type, count in sorted(stage_counts.items()):
        print(f"  {STAGE_SHORT.get(stage_type, stage_type)}: {count} epochs")


if __name__ == "__main__":
    subject_ids = find_subject_ids(EDF_DIR)
    if subject_ids:
        visualize_subject(EDF_DIR, RML_DIR, subject_ids[0])
    else:
        print("No subjects found!")
