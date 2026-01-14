#!/usr/bin/env python3
"""
수면 단계 추론 스크립트
- 일반 오디오 파일 (WAV, MP3 등) 지원
- EDF 파일 지원
- 10시간+ 긴 파일 처리 가능
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import torch
import numpy as np
from pathlib import Path

from sleep_stage_classifier.models.classifier import get_model
from sleep_stage_classifier.config import AudioConfig

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#3498DB", 3: "#9B59B6", 4: "#2ECC71"}


def load_audio_file(audio_path: str):
    """WAV, MP3 등 일반 오디오 파일 로드"""
    import scipy.io.wavfile as wav
    
    ext = Path(audio_path).suffix.lower()
    
    if ext == ".wav":
        sr, audio = wav.read(audio_path)
        # 스테레오 -> 모노
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        # 정규화
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        return audio, sr
    
    elif ext == ".mp3":
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_mp3(audio_path)
            sr = sound.frame_rate
            audio = np.array(sound.get_array_of_samples(), dtype=np.float32)
            if sound.channels == 2:
                audio = audio.reshape((-1, 2)).mean(axis=1)
            audio = audio / 32768.0
            return audio, sr
        except ImportError:
            print("MP3 지원을 위해 pydub 설치 필요: pip install pydub")
            sys.exit(1)
    
    else:
        raise ValueError(f"지원하지 않는 형식: {ext} (WAV, MP3 지원)")


def load_edf_file(edf_path: str):
    """EDF 파일에서 Mic/Snore 채널 로드"""
    from sleep_stage_classifier.data.edf_reader import EDFReader
    reader = EDFReader()
    audio, sr = reader.read_mic_channel(edf_path)
    return audio, sr


def extract_features(audio: np.ndarray, sr: int, epoch_duration: int = 30):
    """오디오에서 Mel-spectrogram 특성 추출"""
    from sleep_stage_classifier.features.audio_features import get_feature_extractor
    
    extractor = get_feature_extractor(
        use_librosa=False,
        sample_rate=sr,
        n_fft=min(512, sr),
        hop_length=min(128, sr // 4)
    )
    
    mel_features, _ = extractor.extract_epoch_features(audio, epoch_duration)
    return mel_features


def load_model(model_path: str, num_classes: int = 5, device: str = "cpu"):
    """학습된 모델 로드"""
    model = get_model(model_type="cnn", num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_batch(model, features: np.ndarray, device: str = "cpu", batch_size: int = 64):
    """배치 단위로 추론 (메모리 효율적)"""
    n_samples = len(features)
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = features[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
            
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)


def format_time(seconds: float) -> str:
    """초를 HH:MM:SS 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_results_csv(predictions, probabilities, output_path: str, epoch_duration: int = 30):
    """결과를 CSV 파일로 저장"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더
        header = ['epoch', 'start_time', 'end_time', 'stage', 'stage_name', 
                  'confidence', 'prob_wake', 'prob_n1', 'prob_n2', 'prob_n3', 'prob_rem']
        writer.writerow(header)
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            start_sec = i * epoch_duration
            end_sec = start_sec + epoch_duration
            
            row = [
                i,
                format_time(start_sec),
                format_time(end_sec),
                pred,
                STAGE_NAMES.get(pred, f"Unknown_{pred}"),
                f"{prob[pred]*100:.1f}%",
                f"{prob[0]*100:.1f}%" if len(prob) > 0 else "N/A",
                f"{prob[1]*100:.1f}%" if len(prob) > 1 else "N/A",
                f"{prob[2]*100:.1f}%" if len(prob) > 2 else "N/A",
                f"{prob[3]*100:.1f}%" if len(prob) > 3 else "N/A",
                f"{prob[4]*100:.1f}%" if len(prob) > 4 else "N/A",
            ]
            writer.writerow(row)
    
    print(f"CSV 저장: {output_path}")


def save_hypnogram(predictions, output_path: str, epoch_duration: int = 30):
    """수면 단계 그래프 (Hypnogram) 저장"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("시각화를 위해 matplotlib 설치 필요: pip install matplotlib")
        return
    
    n_epochs = len(predictions)
    total_hours = (n_epochs * epoch_duration) / 3600
    
    # 시간 축 (시간 단위)
    times = np.arange(n_epochs) * epoch_duration / 3600
    
    # Y축: 수면 단계 (Wake=4, N1=3, N2=2, N3=1, REM=0 으로 뒤집기)
    stage_to_y = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}  # Wake, N1, N2, N3, REM
    y_values = [stage_to_y.get(p, 2) for p in predictions]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- 상단: Hypnogram ---
    ax1 = axes[0]
    
    # 색상으로 채우기
    for i in range(len(predictions) - 1):
        stage = predictions[i]
        color = STAGE_COLORS.get(stage, "#888888")
        ax1.fill_between([times[i], times[i+1]], [y_values[i], y_values[i]], 
                         [y_values[i+1], y_values[i+1]], color=color, alpha=0.7, step='post')
    
    ax1.step(times, y_values, where='post', color='black', linewidth=0.5)
    
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
    ax1.set_xlim(0, total_hours)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Sleep Stage')
    ax1.set_title(f'Sleep Hypnogram ({total_hours:.1f} hours, {n_epochs} epochs)')
    ax1.grid(True, alpha=0.3)
    
    # 범례
    legend_patches = [mpatches.Patch(color=STAGE_COLORS[i], label=STAGE_NAMES[i]) 
                      for i in sorted(STAGE_COLORS.keys())]
    ax1.legend(handles=legend_patches, loc='upper right', ncol=5)
    
    # --- 하단: 수면 단계 분포 ---
    ax2 = axes[1]
    
    unique, counts = np.unique(predictions, return_counts=True)
    stage_counts = {STAGE_NAMES.get(u, f"S{u}"): c for u, c in zip(unique, counts)}
    
    # 모든 단계 포함
    all_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    values = [stage_counts.get(s, 0) for s in all_stages]
    colors = [STAGE_COLORS[i] for i in range(5)]
    percentages = [v / n_epochs * 100 for v in values]
    
    bars = ax2.bar(all_stages, percentages, color=colors, edgecolor='black')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Sleep Stage Distribution')
    ax2.set_ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 100)
    
    # 퍼센트 표시
    for bar, pct, val in zip(bars, percentages, values):
        if pct > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{pct:.1f}%\n({val})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Hypnogram 저장: {output_path}")


def print_summary(predictions, probabilities, total_time: float, epoch_duration: int = 30):
    """결과 요약 출력"""
    n_epochs = len(predictions)
    total_hours = (n_epochs * epoch_duration) / 3600
    
    print("\n" + "=" * 60)
    print("  수면 단계 분석 결과")
    print("=" * 60)
    
    print(f"\n  총 길이: {format_time(n_epochs * epoch_duration)} ({total_hours:.1f}시간)")
    print(f"  총 에포크: {n_epochs}개 (각 {epoch_duration}초)")
    print(f"  처리 시간: {total_time:.1f}초")
    
    print("\n  --- 수면 단계 분포 ---")
    unique, counts = np.unique(predictions, return_counts=True)
    
    # 시간 계산
    for stage, count in sorted(zip(unique, counts)):
        stage_name = STAGE_NAMES.get(stage, f"Unknown_{stage}")
        pct = count / n_epochs * 100
        minutes = count * epoch_duration / 60
        hours = minutes / 60
        
        if hours >= 1:
            time_str = f"{hours:.1f}시간"
        else:
            time_str = f"{minutes:.0f}분"
        
        print(f"    {stage_name:>6}: {count:>4}개 ({pct:>5.1f}%) = {time_str}")
    
    # 평균 신뢰도
    confidences = [prob[pred] for pred, prob in zip(predictions, probabilities)]
    avg_conf = np.mean(confidences) * 100
    print(f"\n  평균 신뢰도: {avg_conf:.1f}%")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="수면 단계 추론")
    parser.add_argument("--input", "-i", type=str, help="입력 오디오 파일 (WAV, MP3, EDF)")
    parser.add_argument("--model", "-m", type=str, default="./output/sleep_stage_cnn.pt", 
                        help="모델 파일 경로")
    parser.add_argument("--output", "-o", type=str, default="./output", 
                        help="결과 저장 폴더")
    parser.add_argument("--num_classes", "-n", type=int, default=5, 
                        help="클래스 수 (기본: 5)")
    parser.add_argument("--device", "-d", type=str, default="auto",
                        help="디바이스 (auto/cpu/cuda)")
    parser.add_argument("--no-plot", action="store_true", help="그래프 생성 안 함")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not args.input:
        print("사용법: python predict.py -i <오디오파일>")
        print("예시:")
        print("  python predict.py -i sleep_recording.wav")
        print("  python predict.py -i recording.mp3")
        print("  python predict.py -i data.edf")
        return
    
    if not os.path.exists(args.input):
        print(f"파일을 찾을 수 없음: {args.input}")
        return
    
    # 모델 확인
    if not os.path.exists(args.model):
        print(f"모델 파일을 찾을 수 없음: {args.model}")
        print("먼저 학습을 실행하세요: python run_pipeline.py")
        return
    
    # 출력 폴더 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"디바이스: {device}")
    
    # 시작
    start_time = time.time()
    
    # 1. 오디오 로드
    print(f"\n[1/3] 오디오 로드 중: {args.input}")
    ext = Path(args.input).suffix.lower()
    
    if ext == ".edf":
        audio, sr = load_edf_file(args.input)
    else:
        audio, sr = load_audio_file(args.input)
    
    duration_sec = len(audio) / sr
    print(f"  길이: {format_time(duration_sec)} ({duration_sec/3600:.1f}시간)")
    print(f"  샘플레이트: {sr}Hz")
    
    # 2. 특성 추출
    print("\n[2/3] 특성 추출 중...")
    features = extract_features(audio, sr, epoch_duration=30)
    print(f"  에포크 수: {len(features)}")
    print(f"  특성 shape: {features.shape}")
    
    # 3. 추론
    print("\n[3/3] 추론 중...")
    model = load_model(args.model, num_classes=args.num_classes, device=device)
    predictions, probabilities = predict_batch(model, features, device=device)
    
    total_time = time.time() - start_time
    
    # 결과 출력
    print_summary(predictions, probabilities, total_time)
    
    # 결과 저장
    base_name = Path(args.input).stem
    
    # CSV 저장
    csv_path = os.path.join(args.output, f"{base_name}_results.csv")
    save_results_csv(predictions, probabilities, csv_path)
    
    # Hypnogram 저장
    if not args.no_plot:
        plot_path = os.path.join(args.output, f"{base_name}_hypnogram.png")
        save_hypnogram(predictions, plot_path)
    
    print(f"\n완료! 결과 폴더: {args.output}")


if __name__ == "__main__":
    main()
