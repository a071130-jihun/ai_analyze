#!/usr/bin/env python3
"""
스마트워치 수면 데이터와 모델 예측 비교
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from sleep_stage_classifier.models.classifier import get_model
from sleep_stage_classifier.features.audio_features import get_feature_extractor


HEALTH_CONNECT_STAGES = {
    1: "Awake",
    2: "Light",
    3: "Deep", 
    4: "REM",
    5: "Awake(OOB)",
    6: "Sleeping",
}

MODEL_STAGES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

HC_TO_MODEL = {
    1: 0,
    2: 1,
    3: 3,
    4: 4,
    5: 0,
    6: 2,
}


def convert_m4a_to_wav(m4a_path: str) -> tuple:
    """M4A를 WAV로 변환하고 numpy array 반환"""
    import scipy.io.wavfile as wav
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    cmd = ["ffmpeg", "-y", "-i", m4a_path, "-ar", "16000", "-ac", "1", tmp_path]
    subprocess.run(cmd, capture_output=True, check=True)
    
    sr, audio = wav.read(tmp_path)
    audio = audio.astype(np.float32) / 32768.0
    
    os.unlink(tmp_path)
    return audio, sr


def get_sleep_session(db_path: str, session_id: int = None):
    """수면 세션 및 단계 데이터 가져오기"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if session_id is None:
        cursor.execute("""
            SELECT row_id, start_time, end_time 
            FROM sleep_session_record_table 
            ORDER BY start_time DESC LIMIT 1
        """)
        row = cursor.fetchone()
        session_id = row[0]
        session_start = row[1]
        session_end = row[2]
    else:
        cursor.execute("""
            SELECT start_time, end_time 
            FROM sleep_session_record_table 
            WHERE row_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        session_start = row[0]
        session_end = row[1]
    
    cursor.execute("""
        SELECT stage_start_time, stage_end_time, stage_type
        FROM sleep_stages_table
        WHERE parent_key = ?
        ORDER BY stage_start_time
    """, (session_id,))
    
    stages = cursor.fetchall()
    conn.close()
    
    return {
        "session_id": session_id,
        "start_time": session_start,
        "end_time": session_end,
        "stages": stages
    }


def create_epoch_labels(stages: list, audio_duration_sec: float, 
                        epoch_duration: int = 30, audio_start_offset: int = 0):
    """스마트워치 단계를 30초 에포크 라벨로 변환"""
    n_epochs = int(audio_duration_sec // epoch_duration)
    labels = np.full(n_epochs, -1, dtype=np.int32)
    
    if not stages:
        return labels
    
    session_start_ms = stages[0][0]
    audio_start_ms = session_start_ms + (audio_start_offset * 1000)
    
    for epoch_idx in range(n_epochs):
        epoch_start_ms = audio_start_ms + (epoch_idx * epoch_duration * 1000)
        epoch_end_ms = epoch_start_ms + (epoch_duration * 1000)
        epoch_mid_ms = (epoch_start_ms + epoch_end_ms) // 2
        
        for stage_start, stage_end, stage_type in stages:
            if stage_start <= epoch_mid_ms < stage_end:
                labels[epoch_idx] = HC_TO_MODEL.get(stage_type, -1)
                break
    
    return labels


def extract_features(audio: np.ndarray, sr: int, epoch_duration: int = 30):
    """Mel-spectrogram 특성 추출"""
    extractor = get_feature_extractor(
        use_librosa=False,
        sample_rate=sr,
        n_fft=min(512, sr),
        hop_length=min(128, sr // 4)
    )
    mel_features, _ = extractor.extract_epoch_features(audio, epoch_duration)
    return mel_features


def load_model(model_path: str, num_classes: int = 5, device: str = "cpu"):
    """모델 로드"""
    model = get_model(model_type="cnn", num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(model, features: np.ndarray, device: str = "cpu"):
    """예측 수행"""
    with torch.no_grad():
        tensor = torch.FloatTensor(features).unsqueeze(1).to(device)
        outputs = model(tensor)
        predictions = outputs.argmax(dim=1).cpu().numpy()
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    return predictions, probabilities


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """정확도 및 지표 계산"""
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    from sklearn.metrics import classification_report, confusion_matrix
    
    valid_mask = y_true >= 0
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        return None
    
    acc = accuracy_score(y_true_valid, y_pred_valid)
    f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true_valid, y_pred_valid)
    
    present_classes = sorted(set(y_true_valid) | set(y_pred_valid))
    target_names = [MODEL_STAGES.get(i, f"Class_{i}") for i in present_classes]
    
    report = classification_report(
        y_true_valid, y_pred_valid, 
        labels=present_classes,
        target_names=target_names,
        zero_division=0
    )
    
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=present_classes)
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "kappa": kappa,
        "report": report,
        "confusion_matrix": cm,
        "classes": present_classes,
        "n_valid": len(y_true_valid),
        "n_total": len(y_true)
    }


def save_comparison_plot(y_true, y_pred, output_path: str, epoch_duration: int = 30):
    """비교 그래프 저장"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib 필요: pip install matplotlib")
        return
    
    n_epochs = len(y_true)
    times = np.arange(n_epochs) * epoch_duration / 3600
    
    stage_to_y = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, -1: 2.5}
    colors = {0: "#E74C3C", 1: "#F39C12", 2: "#3498DB", 3: "#9B59B6", 4: "#2ECC71"}
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    ax1 = axes[0]
    y_smartwatch = [stage_to_y.get(y, 2.5) for y in y_true]
    for i in range(len(y_true) - 1):
        if y_true[i] >= 0:
            ax1.fill_between([times[i], times[i+1]], 
                           [y_smartwatch[i], y_smartwatch[i]], 
                           [y_smartwatch[i+1], y_smartwatch[i+1]], 
                           color=colors.get(y_true[i], "#888"), alpha=0.7, step='post')
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
    ax1.set_xlim(0, times[-1])
    ax1.set_title('Smartwatch (Ground Truth)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    y_model = [stage_to_y.get(y, 2.5) for y in y_pred]
    for i in range(len(y_pred) - 1):
        ax2.fill_between([times[i], times[i+1]], 
                        [y_model[i], y_model[i]], 
                        [y_model[i+1], y_model[i+1]], 
                        color=colors.get(y_pred[i], "#888"), alpha=0.7, step='post')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
    ax2.set_xlim(0, times[-1])
    ax2.set_title('Model Prediction')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    matches = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    valid_mask = y_true >= 0
    
    for i in range(len(matches)):
        if valid_mask[i]:
            color = "#2ECC71" if matches[i] else "#E74C3C"
            ax3.bar(times[i], 1, width=epoch_duration/3600, color=color, alpha=0.7)
    
    ax3.set_xlim(0, times[-1])
    ax3.set_ylim(0, 1.2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_title('Match (Green=Correct, Red=Wrong)')
    ax3.set_yticks([])
    
    legend_patches = [
        mpatches.Patch(color="#2ECC71", label="Correct"),
        mpatches.Patch(color="#E74C3C", label="Wrong"),
    ]
    ax3.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"비교 그래프 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="스마트워치 vs 모델 비교")
    parser.add_argument("--audio", "-a", type=str, default="수면1.m4a")
    parser.add_argument("--db", "-db", type=str, default="health_connect_export.db")
    parser.add_argument("--model", "-m", type=str, default="./output/sleep_stage_cnn.pt")
    parser.add_argument("--session", "-s", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, 
                        help="Audio start offset from session start (seconds)")
    parser.add_argument("--output", "-o", type=str, default="./output")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\n[1/5] 오디오 변환 중...")
    start_time = time.time()
    audio, sr = convert_m4a_to_wav(args.audio)
    audio_duration = len(audio) / sr
    print(f"  길이: {audio_duration/3600:.2f}시간 ({int(audio_duration)}초)")
    print(f"  샘플레이트: {sr}Hz")
    
    print("\n[2/5] 스마트워치 데이터 로드 중...")
    session = get_sleep_session(args.db, args.session)
    print(f"  세션 ID: {session['session_id']}")
    start_dt = datetime.fromtimestamp(session['start_time'] / 1000)
    end_dt = datetime.fromtimestamp(session['end_time'] / 1000)
    print(f"  수면 시간: {start_dt.strftime('%H:%M')} ~ {end_dt.strftime('%H:%M')}")
    print(f"  수면 단계 수: {len(session['stages'])}")
    
    print("\n[3/5] 특성 추출 중...")
    features = extract_features(audio, sr)
    n_epochs = len(features)
    print(f"  에포크 수: {n_epochs}")
    
    print("\n[4/5] 모델 예측 중...")
    model = load_model(args.model, num_classes=5, device=device)
    predictions, probabilities = predict(model, features, device)
    
    print("\n[5/5] 라벨 비교 중...")
    labels = create_epoch_labels(
        session['stages'], 
        audio_duration, 
        epoch_duration=30,
        audio_start_offset=args.offset
    )
    
    valid_count = np.sum(labels >= 0)
    print(f"  매칭된 라벨: {valid_count}/{n_epochs} ({100*valid_count/n_epochs:.1f}%)")
    
    metrics = calculate_metrics(labels, predictions)
    
    print("\n" + "=" * 60)
    print("  평가 결과 (스마트워치 vs 모델)")
    print("=" * 60)
    
    if metrics:
        print(f"\n  유효 에포크: {metrics['n_valid']}/{metrics['n_total']}")
        print(f"\n  Accuracy:     {metrics['accuracy']*100:.1f}%")
        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):{metrics['f1_weighted']:.4f}")
        print(f"  Cohen's Kappa:{metrics['kappa']:.4f}")
        
        kappa = metrics['kappa']
        if kappa < 0.20:
            kappa_level = "Poor"
        elif kappa < 0.40:
            kappa_level = "Fair"
        elif kappa < 0.60:
            kappa_level = "Moderate"
        elif kappa < 0.80:
            kappa_level = "Substantial"
        else:
            kappa_level = "Almost Perfect"
        print(f"  Kappa 등급:   {kappa_level}")
        
        print(f"\n  Classification Report:\n{metrics['report']}")
        
        save_comparison_plot(
            labels, predictions,
            os.path.join(args.output, "smartwatch_comparison.png")
        )
    else:
        print("  매칭된 라벨이 없습니다. --offset 옵션을 조정해보세요.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
